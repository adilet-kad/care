#!/usr/bin/env bash
# reproduce.sh -- regenerate every table and figure in the paper from the released logs.
#
# One command, no GPU, no API key, no external cleaner. The proposer logs in csvs/ are
# the released artifact; everything downstream is replayed from them. Regenerating the
# LOGS themselves needs the cleaners' own repos and is a separate, much longer path --
# see docs/REPRODUCE.md section 3.
#
#     ./reproduce.sh            # sweeps, audits, figures, tables, then verify  (~1 h)
#     ./reproduce.sh --quick    # skip everything on tax (minutes instead of an hour)
#     ./reproduce.sh --verify   # only re-check that the stored results still regenerate
#
# Exit status is non-zero if any stage fails, so this is usable in CI.

set -euo pipefail
cd "$(dirname "$0")"

QUICK=0; ONLY_VERIFY=0
for a in "$@"; do
  case "$a" in
    --quick)  QUICK=1 ;;
    --verify) ONLY_VERIFY=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

PY=${PYTHON:-python}
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# The venv is not activated for you: doing so silently from a script makes it hard to
# tell which interpreter produced a number. Fail loudly instead.
$PY -c 'import care' 2>/dev/null || {
  echo "CARE is not importable. Activate the venv first:" >&2
  echo "    python3 -m venv .venv && source .venv/bin/activate" >&2
  echo "    pip install -r requirements.txt && pip install -e ." >&2
  exit 1
}

# Fixed everywhere; changing --alphas or --seeds breaks comparability with the paper.
A="--experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 --fast-verify on"
P="--experiment poison --poison-fracs 0.0 0.05 0.1 0.2 --seeds 10 --fast-verify on"
SMALL="hospital beers flights rayyan"
run() { $PY -m bench.run_study "$@"; }

if [ "$ONLY_VERIFY" = 0 ]; then
  step "1/8  Table 1: each proposer under its own protocol (oracle detection)"
  for ds in $SMALL; do
    run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv"         $A --scoring errors   --out experiments/results_baran_errors
    run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv"         $A --scoring detected --out experiments/results_baran_detected
    run --dataset $ds --backends "jellyfish:csvs/jellyfish_${ds}_mapped.csv" $A --scoring detected --out experiments/results_detected
    run --dataset $ds --backends "gpt4o:csvs/gpt4o_${ds}_mapped.csv"         $A --scoring detected --out experiments/results_gpt4o
  done
  for ds in hospital beers flights; do
    run --dataset $ds --backends "bclean:csvs/bclean_${ds}_mapped.csv"             $A --scoring errors   --out experiments/results_bclean
    run --dataset $ds --backends "bclean_keep:csvs/bclean_keep_${ds}_mapped.csv"   $A --scoring detected --out experiments/results_bclean_keep_oracle
    run --dataset $ds --backends "holoclean:csvs/holoclean_${ds}_mapped.csv"       $A --scoring errors   --out experiments/results_holoclean
    run --dataset $ds --backends "holoclean_keep:csvs/holoclean_keep_${ds}_mapped.csv" $A --scoring detected --out experiments/results_holoclean_keep_oracle
  done
  run --dataset flights --backends "retclean:csvs/retclean_flights_mapped.csv" $A --scoring detected --out experiments/results_retclean

  step "2/8  Table 3: the same logs behind a real detector (Raha) and a constraint detector"
  for ds in $SMALL; do
    run --dataset $ds --backends "baran_raha:csvs/baran_raha_${ds}_mapped.csv" $A --scoring detected \
        --detection log:csvs/raha_${ds}_detected.csv --out experiments/results_raha_baran
    run --dataset $ds --backends "jellyfish:csvs/jellyfish_${ds}_mapped.csv"   $A --scoring detected \
        --detection log:csvs/raha_${ds}_detected.csv --out experiments/results_raha_jelly
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
  run --dataset flights --backends "jellyfish:csvs/jellyfish_flights_mapped.csv" $A --scoring detected \
      --detection constraints --out experiments/results_constraints_jelly

  step "3/8  Table 2: Theorems 1 and 2 (untrusted and trusted corruption)"
  poison() {  # dataset backend alpha scoring outdir
    run --dataset "$1" --backends "$2" $P --alphas "$3" --scoring "$4"                  --out "experiments/$5"
    run --dataset "$1" --backends "$2" $P --alphas "$3" --scoring "$4" --trusted-poison --out "experiments/$5"
  }
  # Table 2 uses the five Baran/BClean configurations; the two Jellyfish ones are run
  # because Section 7.6 reports that they cannot carry a Theorem 2 exhibit (the gate
  # certifies nothing, so no poisoned cell can reach the applied set).
  poison beers    "jellyfish:csvs/jellyfish_beers_mapped.csv"    0.1 detected results_poison_jellyfish_beers
  poison hospital "jellyfish:csvs/jellyfish_hospital_mapped.csv" 0.2 detected results_poison_jellyfish_hospital
  poison hospital "baran:csvs/baran_hospital_mapped.csv"         0.2 errors   results_poison_baran_hospital
  poison beers    "baran:csvs/baran_beers_mapped.csv"            0.2 errors   results_poison_baran_beers
  poison rayyan   "baran:csvs/baran_rayyan_mapped.csv"           0.2 errors   results_poison_baran_rayyan
  poison hospital "bclean:csvs/bclean_hospital_mapped.csv"       0.2 errors   results_poison_bclean_hospital
  poison flights  "bclean:csvs/bclean_flights_mapped.csv"        0.2 errors   results_poison_bclean_flights

  step "4/8  Baran reruns: five extra draws per dataset (Table 1 medians, Section 9)"
  for ds in $SMALL; do
    for i in 1 2 3 4 5; do
      run --dataset $ds --backends "baran:csvs/baran_var_${ds}/$i/baran_${ds}_mapped.csv" \
          $A --scoring errors --out experiments/results_baran_var_${ds}/$i
    done
  done

  if [ "$QUICK" = 0 ]; then
    step "5/8  tax (200K rows, 3M cells): oracle, Raha, Jellyfish shard, six Baran draws"
    run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" $A --scoring errors   --out experiments/results_baran_errors
    run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" $A --scoring detected --out experiments/results_baran_detected
    run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" $A --scoring detected \
        --detection log:csvs/raha_tax_detected.csv --out experiments/results_raha_baran
    # --row-range is mandatory: the Jellyfish tax log covers rows [0,25000) only.
    run --dataset tax --backends "jellyfish:csvs/jellyfish_tax_mapped.csv" $A --scoring detected \
        --row-range 0 25000 --out experiments/results_jelly_tax25k
    for i in 1 2 3 4 5; do
      run --dataset tax --backends "baran:csvs/baran_var_tax/$i/baran_tax_mapped.csv" \
          $A --scoring errors --out experiments/results_baran_var_tax/$i
    done
  else
    echo "  [--quick] skipping tax; its Table 1 / Table 3 rows and Section 7.3-7.5 numbers will be stale"
  fi

  step "6/8  audits"
  $PY audit_proposers.py            # accuracy on error cells, pi_0, harm            -> proposer_accuracy.csv
  $PY audit_baran_median.py         # Table 1's Baran rows as medians over draws     -> baran_median.csv
  $PY audit_alpha_ceiling.py --logs "csvs/baran_tax_mapped.csv" "csvs/baran_var_tax/*/baran_tax_mapped.csv" \
      --per-stratum --out experiments/tax_draws.csv   # per-draw, per-column (Sections 7.2-7.4)
  $PY audit_calibration.py          # is confidence a probability?                  -> calibration.csv
  $PY audit_recalibration.py        # does Platt/isotonic fix it?                    -> recalibration.csv
  $PY audit_heldout_budget.py       # the naive held-out threshold vs the bound      -> heldout_budget.csv
  $PY audit_sensitivity.py          # |Lambda| x delta grid                          -> sensitivity.csv
  $PY audit_label_budget.py         # what certification costs in labels             -> label_budget.csv
  $PY audit_drift_probe.py          # source-split drift probe (Section 9)           -> drift_probe.csv
  # error drop rate, apply-all vs CARE -> error_drop.csv (+ tab5). Note: ${QUICK:+...} would
  # expand for QUICK=0 as well (0 is a non-empty string), so the branch is written out.
  if [ "$QUICK" = 1 ]; then
    $PY audit_error_drop.py --raha --skip-tax --tex paper/tables/tab5_error_drop.tex
  else
    $PY audit_error_drop.py --raha --tex paper/tables/tab5_error_drop.tex
  fi
  # Controlled error-rate sweep. The variant logs ship in csvs/*_ni_*; the ceilings were
  # sealed before calibration and audit_sweep.py checks the seal. The join reads the stored
  # per-variant sweeps in experiments/results_sweep_baran_var/ (regenerating those from the
  # variant logs is the loop in docs/REPRODUCE.md section 3).
  if ls csvs/*_ni_*_mapped.csv >/dev/null 2>&1; then
    $PY audit_sweep.py || echo "  (sweep join incomplete; see docs/REPRODUCE.md section 3)"
  else
    echo "  [skip] no csvs/*_ni_*_mapped.csv -- see docs/REPRODUCE.md section 3"
  fi

  step "7/8  governance cost per repair"
  $PY -m bench.throughput --datasets flights hospital beers --exact-sample 50 --out experiments/throughput.csv

  step "8/8  figures and tables"
  $PY make_figures.py
fi

step "verification"
echo "-- the fast verification path is exact, not approximate"
$PY tools/verify_fastpath_equivalence.py
echo "-- its precondition is checked at run time and its failure is handled"
$PY tools/verify_precondition_fallback.py
echo "-- every stored sweep regenerates from the documented command"
if [ "$QUICK" = 1 ]; then
  $PY tools/verify_results_reproduce.py --only hospital beers flights rayyan
else
  $PY tools/verify_results_reproduce.py
fi

printf '\n\033[1mdone.\033[0m figures in paper/figures, tables in paper/tables\n'
