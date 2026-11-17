#!/bin/bash
# Runs the experiments of the thesis:  bash calm/run_experiments.sh [experiment ...]   (no argument: all five, about 5.5 hours)
#
#   experiment  runs (n = NSL-KDD, u = UNSW-NB15)                                                    seeds   README results
#   main        {n,u}_em: CALM-IDS with the data stream; {n,u}_frozen: its initial model, without it    10     1, 2, 3, 5
#   online      {n,u}_enc: the variant that also retrains the encoder on the data stream                 5     3
#   zero_day    {n,u}_<category>_{a,b}: one attack category held out, in setting A or B;                 5     4
#               {n,u}_{DoS,Generic}_b_nocentre: setting B without the online update
#   ablation    {n,u}_{nosn,k1,noopen}: without spectral normalisation, one Gaussian per class,           5     5
#               without the central component
#   curves      {n,u}_em: CALM-IDS with the test pAUC recorded every 5 batches                          5     3
#
# Output: results/<experiment>/<run>.json (one record per seed), the test-set scores in results/<experiment>/dumps/, logs in
# results/logs/. Finished runs are skipped, so the script can be stopped and restarted.
# Optional: PY (Python interpreter, default python), NPROC (runs in parallel, default 2), SEEDS (number of seeds).
set -u; cd "$(dirname "$0")"
EXPERIMENTS=${*:-main online zero_day ablation curves}
for E in $EXPERIMENTS; do case $E in main|online|zero_day|ablation|curves) ;; *) echo "unknown experiment: $E"; exit 1 ;; esac; done
PY=${PY:-python}; NPROC=${NPROC:-2}; export OMP_NUM_THREADS=2; L=../results/logs; mkdir -p "$L"; PIDS=""
for E in $EXPERIMENTS; do
  O=../results/$E; N=${SEEDS:-5}; DUMP="--dump $O/dumps"; JOBS=()
  for ds in unsw nsl; do d=${ds:0:1}; C="--dataset $ds"
    if [ $ds = nsl ]; then CATS="DoS Probe R2L U2R"; W=DoS
    else CATS="Generic Exploits Fuzzers DoS Reconnaissance Analysis Backdoor Shellcode Worms"; W=Generic; fi
    case $E in
      main)     N=${SEEDS:-10}; JOBS+=("${d}_em|$C --mode online" "${d}_frozen|$C --mode offline") ;;
      online)   JOBS+=("${d}_enc|$C --enc_online 1") ;;
      zero_day) for c in $CATS; do JOBS+=("${d}_${c}_a|$C --lofo $c --lofo_mode a" "${d}_${c}_b|$C --lofo $c --lofo_mode b"); done
                JOBS+=("${d}_${W}_b_nocentre|$C --lofo $W --lofo_mode b --centre_update 0") ;;
      ablation) JOBS+=("${d}_nosn|$C --c 0 --c0 0" "${d}_k1|$C --K 1 --J 1" "${d}_noopen|$C --eps 1e-6") ;;
      curves)   DUMP=""; JOBS+=("${d}_em|$C --eval_every 5") ;;
    esac
  done
  mkdir -p "$O"; echo "[calm] $E: ${#JOBS[@]} runs x $N seeds, started $(date '+%m-%d %H:%M:%S')"
  for J in "${JOBS[@]}"; do
    TAG=${J%%|*}; ARG=${J#*|}
    [ -f "$O/$TAG.json" ] && { echo "  skip $TAG"; continue; }
    while :; do                                                  # wait until fewer than NPROC runs are active
      alive=""; for p in $PIDS; do kill -0 $p 2>/dev/null && alive="$alive $p"; done
      PIDS=$alive; [ $(echo $PIDS | wc -w) -lt $NPROC ] && break; sleep 5
    done
    echo "  launch $TAG ($(date '+%H:%M:%S'))"
    nice -n 10 $PY -u calm_ids.py $ARG --seed0 5009 --seeds $N $DUMP --tag "$TAG" --out "$O" > "$L/${E}_$TAG.log" 2>&1 &
    PIDS="$PIDS $!"; sleep 2
  done
done
wait; echo "[calm] done $(date '+%m-%d %H:%M:%S')"
