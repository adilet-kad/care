#!/usr/bin/env bash
# reproduce.sh -- regenerate every result, table and figure from the cleaner logs in csvs/.
#
# CPU only; no GPU, API key or cleaner code is needed. Regenerating the logs themselves
# requires each cleaner's repository (docs/REPRODUCE.md, section 3).
#
#     ./reproduce.sh            # sweeps, audits, figures, verification
#     ./reproduce.sh --quick    # skip tax (about 16 GB of RAM) and the error-rate sweep
#     ./reproduce.sh --verify   # only check that stored sweeps regenerate
#     ./reproduce.sh --from 6   # resume at step 6 (earlier steps' results are kept)
#
# Exits non-zero if any stage fails.

set -euo pipefail
cd "$(dirname "$0")"

QUICK=0; ONLY_VERIFY=0; FROM=1; prev=""
for a in "$@"; do
  if [ "$prev" = "--from" ]; then FROM="$a"; prev=""; continue; fi
  case "$a" in
    --quick)  QUICK=1 ;;
    --verify) ONLY_VERIFY=1 ;;
    --from)   prev="--from" ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

PY_REAL=${PYTHON:-python}; PY=$PY_REAL
step() {
  local n=${1%%/*}
  if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -lt "$FROM" ]; then
    printf '\n== %s  (skipped: --from %s)\n' "$1" "$FROM"; PY=:
  else
    printf '\n== %s\n' "$1"; PY=$PY_REAL
  fi
}
$PY_REAL -c 'import care' 2>/dev/null || {
  echo "CARE is not importable; run: pip install -r requirements.txt && pip install -e ." >&2
  exit 1
}

A="--experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 --fast-verify on"
P="--experiment poison --poison-fracs 0.0 0.05 0.1 0.2 --seeds 10 --fast-verify on"
SMALL="hospital beers flights rayyan"
run() { $PY -m bench.run_study "$@"; }

if [ "$ONLY_VERIFY" = 0 ]; then
  step "1/9  oracle detection: the six cleaners (Table 3)"
  for ds in $SMALL; do
    run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv"   $A --scoring errors   --out experiments/results_baran_errors
    run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv"   $A --scoring detected --out experiments/results_baran_detected
    run --dataset $ds --backends "gidcl:csvs/gidcl_${ds}_mapped.csv"   $A --scoring errors   --out experiments/results_gidcl
    run --dataset $ds --backends "zeroec:csvs/zeroec_${ds}_mapped.csv" $A --scoring errors   --out experiments/results_zeroec
  done
  for ds in hospital beers flights; do
    run --dataset $ds --backends "bclean:csvs/bclean_${ds}_mapped.csv"                 $A --scoring errors   --out experiments/results_bclean
    run --dataset $ds --backends "bclean_keep:csvs/bclean_keep_${ds}_mapped.csv"       $A --scoring detected --out experiments/results_bclean_keep_oracle
    run --dataset $ds --backends "bclean_plus:csvs/bclean_plus_${ds}_mapped.csv"       $A --scoring errors   --out experiments/results_bclean_plus
    run --dataset $ds --backends "holoclean:csvs/holoclean_${ds}_mapped.csv"           $A --scoring errors   --out experiments/results_holoclean
    run --dataset $ds --backends "holoclean_keep:csvs/holoclean_keep_${ds}_mapped.csv" $A --scoring detected --out experiments/results_holoclean_keep_oracle
  done
  for ds in hospital beers rayyan; do
    run --dataset $ds --backends "gidcl_scored:csvs/gidcl_scored_${ds}_mapped.csv"       $A --scoring errors --out experiments/results_gidcl_scored
    run --dataset $ds --backends "gidcl_scored_s1:csvs/gidcl_scored_s1_${ds}_mapped.csv" $A --scoring errors --out experiments/results_gidcl_scored_s1
    run --dataset $ds --backends "gidcl_scored_s2:csvs/gidcl_scored_s2_${ds}_mapped.csv" $A --scoring errors --out experiments/results_gidcl_scored_s2
  done

  step "2/9  real detectors: Raha, GIDCL's own, a constraint detector (Table 6)"
  for ds in $SMALL; do
    run --dataset $ds --backends "baran_raha:csvs/baran_raha_${ds}_mapped.csv" $A --scoring detected \
        --detection log:csvs/raha_${ds}_detected.csv --out experiments/results_raha_baran
    run --dataset $ds --backends "gidcl:csvs/gidcl_${ds}_mapped.csv" $A --scoring detected \
        --detection log:csvs/gidcl_${ds}_detected.csv --out experiments/results_gidcl_owndet
  done
  for ds in hospital beers flights; do
    run --dataset $ds --backends "bclean_keep:csvs/bclean_keep_${ds}_mapped.csv"       $A --scoring detected \
        --detection log:csvs/raha_${ds}_detected.csv --out experiments/results_raha_bclean
    run --dataset $ds --backends "holoclean_keep:csvs/holoclean_keep_${ds}_mapped.csv" $A --scoring detected \
        --detection log:csvs/raha_${ds}_detected.csv --out experiments/results_raha_holoclean
  done
  for ds in hospital flights; do
    run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv" $A --scoring detected \
        --detection constraints --out experiments/results_constraints_baran
  done

  step "3/9  corruption through untrusted and trusted sources (Section 7.6)"
  poison() {  # dataset backend outdir
    run --dataset "$1" --backends "$2" $P --alphas 0.2 --scoring errors                  --out "experiments/$3"
    run --dataset "$1" --backends "$2" $P --alphas 0.2 --scoring errors --trusted-poison --out "experiments/$3"
  }
  poison hospital "baran:csvs/baran_hospital_mapped.csv"   results_poison_baran_hospital
  poison beers    "baran:csvs/baran_beers_mapped.csv"      results_poison_baran_beers
  poison rayyan   "baran:csvs/baran_rayyan_mapped.csv"     results_poison_baran_rayyan
  poison hospital "bclean:csvs/bclean_hospital_mapped.csv" results_poison_bclean_hospital
  poison flights  "bclean:csvs/bclean_flights_mapped.csv"  results_poison_bclean_flights

  step "4/9  Baran: five more draws per table"
  for ds in $SMALL; do
    for i in 1 2 3 4 5; do
      run --dataset $ds --backends "baran:csvs/baran_var_${ds}/$i/baran_${ds}_mapped.csv" \
          $A --scoring errors --out experiments/results_baran_var_${ds}/$i
    done
  done

  if [ "$QUICK" = 0 ]; then
    step "5/9  tax (200K rows, 3M cells)"
    run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" $A --scoring errors   --out experiments/results_baran_errors
    run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" $A --scoring detected --out experiments/results_baran_detected
    run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" $A --scoring detected \
        --detection log:csvs/raha_tax_detected.csv --out experiments/results_raha_baran
    run --dataset tax --backends "gidcl:csvs/gidcl_tax_mapped.csv" $A --scoring errors --out experiments/results_gidcl
    run --dataset tax --backends "gidcl:csvs/gidcl_tax_mapped.csv" $A --scoring detected \
        --detection log:csvs/gidcl_tax_detected.csv --out experiments/results_gidcl_owndet
    for i in 1 2 3 4 5; do
      run --dataset tax --backends "baran:csvs/baran_var_tax/$i/baran_tax_mapped.csv" \
          $A --scoring errors --out experiments/results_baran_var_tax/$i
    done

    step "6/9  controlled error-rate sweep (Ni et al. variants)"
    if [ "$PY" != ":" ] && [ ! -f data/ni_variants.csv ]; then
      echo "  importing Ni et al.'s variants (one-time download)"
      [ -d /tmp/adr ] || git clone --depth 1 https://github.com/WelkinNi/Automatic-Data-Repair.git /tmp/adr
      $PY tools/import_ni_variants.py --adr /tmp/adr
      $PY tools/import_ni_variants.py --verify
    fi
    for log in csvs/baran_*_ni_inner*_mapped.csv csvs/baran_var_ni/*/*/baran_*_mapped.csv; do
      v=$(basename "$log" _mapped.csv); v=${v#baran_}
      case "$log" in csvs/baran_var_ni/*) d=$(basename "$(dirname "$log")") ;; *) d=0 ;; esac
      run --dataset "$v" --backends "baran:$log" $A --scoring errors \
          --out "experiments/results_sweep_baran_var/$v/$d"
    done
  else
    echo "  --quick: tax and the error-rate sweep are skipped; their numbers keep the stored values"
  fi

  step "7/9  audits"
  $PY audit_proposers.py
  $PY audit_baran_median.py
  $PY audit_calibration.py
  $PY audit_recalibration.py
  $PY audit_heldout_budget.py
  $PY audit_sensitivity.py
  $PY audit_label_budget.py
  $PY audit_drift_probe.py
  $PY audit_gidcl_confidence.py
  if [ "$QUICK" = 1 ]; then
    $PY audit_error_drop.py --raha --skip-tax --tex paper/tables/extra_error_drop.tex
  else
    $PY audit_alpha_ceiling.py --logs "csvs/baran_tax_mapped.csv" "csvs/baran_var_tax/*/baran_tax_mapped.csv" \
        --per-stratum --out experiments/tax_draws.csv
    $PY audit_error_drop.py --raha --tex paper/tables/extra_error_drop.tex
    $PY audit_sweep.py
    $PY tools/verify_incremental.py tax --sample 60
  fi
  $PY tools/verify_incremental.py hospital --sample 0
  $PY tools/audit_planner.py
  $PY tools/audit_partitions.py
  $PY tools/audit_trust_noise.py
  $PY tools/audit_drift_monitor.py
  for d in 1 2 3; do
    $PY tools/audit_drift_monitor.py --sources flights:10 hospital:10 hospital:70 --draw $d \
        --alphas 0.2 --out experiments/drift_monitor_d$d.csv
  done
  $PY - <<'EOF'
import csv, glob
rows = []
for p in sorted(glob.glob("experiments/drift_monitor_d[123].csv")):
    rows += list(csv.DictReader(open(p)))
with open("experiments/drift_monitor_more.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
EOF
  rm -f experiments/drift_monitor_d[123].csv
  if [ "$QUICK" = 1 ]; then
    echo "  --quick: Table 4 (procedures) and the scaling numbers of Section 4.2 keep the stored values"
  else
    $PY tools/audit_procedures.py --tax     # all 21 scored pairs (Table 4)
    $PY tools/audit_scaling.py              # Section 4.2
  fi
  $PY tools/audit_overwrite_formats.py --all

  step "8/9  cost per repair"
  $PY -m bench.throughput --datasets flights hospital beers --exact-sample 50 --out experiments/throughput.csv

  step "9/9  figures and tables"
  $PY make_figures.py
fi

step "verification"
$PY tools/verify_fastpath_equivalence.py
$PY tools/verify_precondition_fallback.py
$PY tools/verify_gate_first.py
if [ "$QUICK" = 1 ]; then
  $PY tools/verify_results_reproduce.py --only hospital beers flights rayyan
else
  $PY tools/verify_results_reproduce.py
fi

printf '\ndone: figures in paper/figures, tables in paper/tables\n'
