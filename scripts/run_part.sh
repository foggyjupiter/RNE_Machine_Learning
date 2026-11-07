#!/bin/bash
# One share of the experiment suite on a parallel machine (cloud session).
#   scripts/run_part.sh PART            (PART = M0 | M1 | M2 | M3, see rneml/suite.py PARTS)
# Env: DATA_DIR (default data), REAL_DIR (first-study recordings, default real/data-all),
#      RESULTS_VIA=hf (machines that cannot push: results go to the HF dataset experiments/),
#      REVERSE=1 (run the part's list last-first: a helper taking the tail of a slow part),
#      ONLY="name1 name2" (run just these experiments of the part: several machines per part),
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
# RESULTS_VIA=hf: no push access on this machine → results go to the HF dataset (experiments/)
VIA=""; [ "${RESULTS_VIA:-git}" = "hf" ] && VIA="--via-hf"
SETS=$(python3 -c "from rneml.suite import PART_DATA, SUITE_TO_HF; print(' '.join(SUITE_TO_HF[k] for k in PART_DATA['$PART']))")
[ -d "$R" ] || { echo "real recordings missing at $R" | tee -a $LOG; exit 1; }
if [ "${SKIP_FETCH:-0}" != "1" ]; then
  echo "fetch $SETS $(date -u)" >> $LOG
  rneml fetch --dir "$D" --sets $SETS >> $LOG 2>&1 || { echo "FETCH FAILED" >> $LOG; exit 1; }
fi
# suite "v1" (main data) = dataset v3; the old v1 is the repository root
V1=${V1_DIR:-$D/v3}; V1OLD=${V1OLD_DIR:-$D}; V1R2=${V1R2_DIR:-$D/v1_r2}; BB=${BB_DIR:-$D/bb_fixed}; MR=${MR_DIR:-$D/multi_rpm}
V1SIG=${V1SIG_DIR:-$D/v1_sig}; V1CLK=${V1CLK_DIR:-$D/v1_clk}; V3NOSIG=${V3NOSIG_DIR:-$D/v3_nosig}; MRNOSIG=${MRNOSIG_DIR:-$D/multi_rpm_nosig}
if [ "$PART" = "M0" ] && [ "${COMPONENTS:-1}" = "1" ]; then
  echo "COMPONENTS START $(date -u)" >> $LOG
  rneml components --data-dir "$V1" --real-dir "$R" --workers "$(nproc)" >> $LOG 2>&1
fi
echo "SUITE PART $PART START $(date -u)" >> $LOG
rneml suite --part "$PART" --v1 "$V1" --v1-old "$V1OLD" --v1-r2 "$V1R2" --bb-fixed "$BB" --multi-rpm "$MR" \
  --data "v1_sig=$V1SIG" --data "v1_clk=$V1CLK" --data "v3_nosig=$V3NOSIG" --data "multi_rpm_nosig=$MRNOSIG" \
  --real-dir "$R" --size cpu --threads "$(nproc)" --max-priority 5 ${REVERSE:+--reverse} \
  ${ONLY:+--only $ONLY} >> $LOG 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  for i in $(seq 180); do kill -0 $PID 2>/dev/null || break; sleep 10; done
  python3 scripts/push_results.py --part "$PART" --no-models $VIA >> $LOG 2>&1
done
python3 scripts/push_results.py --part "$PART" $VIA >> $LOG 2>&1
echo "PART $PART DONE $(date -u)" >> $LOG
