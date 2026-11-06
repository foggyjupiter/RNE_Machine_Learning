#!/bin/bash
# One share of the experiment suite on a parallel machine (cloud session).
#   scripts/run_part.sh PART            (PART = M0 | M1 | M2 | M3, see rneml/suite.py PARTS)
# Env: DATA_DIR (default data), REAL_DIR (first-study recordings, default real/data-all),
#      COMPONENTS (M0 only: run the component analysis first; default 1).
# Fetches only the data the part needs, runs it (resumable), and every 30 min and at the
# end sends the results back (scripts/push_results.py). Writes "PART <P> DONE" to the log.
set -u
PART=$1
D=${DATA_DIR:-data}
R=${REAL_DIR:-real/data-all}
export HF_TOKEN=${HF_TOKEN:-proxy-injected}
mkdir -p logs
LOG=logs/suite_$PART.log
SETS=$(python3 -c "from rneml.suite import PART_DATA; print(' '.join(PART_DATA['$PART']))")
[ -d "$R" ] || { echo "real recordings missing at $R" | tee -a $LOG; exit 1; }
if [ "${SKIP_FETCH:-0}" != "1" ]; then
  echo "fetch $SETS $(date -u)" >> $LOG
  rneml fetch --dir "$D" --sets $SETS >> $LOG 2>&1 || { echo "FETCH FAILED" >> $LOG; exit 1; }
fi
V1=${V1_DIR:-$D}; V1R2=${V1R2_DIR:-$D/v1_r2}; BB=${BB_DIR:-$D/bb_fixed}; MR=${MR_DIR:-$D/multi_rpm}
if [ "$PART" = "M0" ] && [ "${COMPONENTS:-1}" = "1" ]; then
  echo "COMPONENTS START $(date -u)" >> $LOG
  rneml components --data-dir "$V1" --real-dir "$R" --workers "$(nproc)" >> $LOG 2>&1
fi
echo "SUITE PART $PART START $(date -u)" >> $LOG
rneml suite --part "$PART" --v1 "$V1" --v1-r2 "$V1R2" --bb-fixed "$BB" --multi-rpm "$MR" \
  --real-dir "$R" --size cpu --threads "$(nproc)" --max-priority 5 >> $LOG 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  for i in $(seq 180); do kill -0 $PID 2>/dev/null || break; sleep 10; done
  python3 scripts/push_results.py --part "$PART" --no-models >> $LOG 2>&1
done
python3 scripts/push_results.py --part "$PART" >> $LOG 2>&1
echo "PART $PART DONE $(date -u)" >> $LOG
