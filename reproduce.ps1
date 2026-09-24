<#
reproduce.ps1 -- regenerate every table and figure in the paper from the released logs.

Windows counterpart of reproduce.sh, step for step and flag for flag, so a number
produced on either platform comes from the same documented command. One command, no GPU,
no API key, no external cleaner. The proposer logs in csvs/ are the released artifact;
everything downstream is replayed from them. Regenerating the LOGS themselves needs the
cleaners' own repos and is a separate, much longer path; see docs/REPRODUCE.md section 3.

    .\reproduce.ps1            # sweeps, audits, figures, tables, then verify  (~1 h)
    .\reproduce.ps1 -Quick     # skip everything on tax (minutes instead of an hour)
    .\reproduce.ps1 -Verify    # only re-check that the stored results still regenerate

Exit status is non-zero if any stage fails, so this is usable in CI.
#>

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Verify,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path | Select-Object -Skip 1 -First 12
    exit 0
}

Set-Location $PSScriptRoot

# Windows consoles default to a legacy code page. Python then raises on any non-ASCII
# character it prints or writes, which would turn a cosmetic message into a failed
# stage; make Python use UTF-8 for stdio and files regardless of the console.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$PY = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Step([string]$Message) {
    Write-Host ""
    Write-Host "== $Message" -ForegroundColor White
}

# Every external command is checked: PowerShell does not stop on a non-zero exit status
# by itself, and a silently failed sweep would leave a stale result file in place.
function Invoke-Checked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Cmd)
    & $Cmd[0] @($Cmd[1..($Cmd.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        Write-Error "command failed (exit $LASTEXITCODE): $($Cmd -join ' ')"
        exit $LASTEXITCODE
    }
}

# The venv is not activated for you: doing so silently from a script makes it hard to
# tell which interpreter produced a number. Fail loudly instead.
& $PY -c "import care" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "CARE is not importable. Activate the venv first:"
    Write-Host "    python -m venv .venv; .\.venv\Scripts\Activate.ps1"
    Write-Host "    pip install -r requirements.txt; pip install -e ."
    exit 1
}

# Fixed everywhere; changing --alphas or --seeds breaks comparability with the paper.
$A = @("--experiment", "pareto", "--alphas", "0.05", "0.1", "0.2", "--seeds", "10", "--fast-verify", "on")
$P = @("--experiment", "poison", "--poison-fracs", "0.0", "0.05", "0.1", "0.2", "--seeds", "10", "--fast-verify", "on")
$SMALL = @("hospital", "beers", "flights", "rayyan")

function Run {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArgs)
    Invoke-Checked $PY -m bench.run_study @RunArgs
}

if (-not $Verify) {
    Step "1/8  Table 1: each proposer under its own protocol (oracle detection)"
    foreach ($ds in $SMALL) {
        Run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv"         @A --scoring errors   --out experiments/results_baran_errors
        Run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv"         @A --scoring detected --out experiments/results_baran_detected
        Run --dataset $ds --backends "jellyfish:csvs/jellyfish_${ds}_mapped.csv" @A --scoring detected --out experiments/results_detected
        Run --dataset $ds --backends "gpt4o:csvs/gpt4o_${ds}_mapped.csv"         @A --scoring detected --out experiments/results_gpt4o
    }
    foreach ($ds in @("hospital", "beers", "flights")) {
        Run --dataset $ds --backends "bclean:csvs/bclean_${ds}_mapped.csv"                 @A --scoring errors   --out experiments/results_bclean
        Run --dataset $ds --backends "bclean_keep:csvs/bclean_keep_${ds}_mapped.csv"       @A --scoring detected --out experiments/results_bclean_keep_oracle
        Run --dataset $ds --backends "holoclean:csvs/holoclean_${ds}_mapped.csv"           @A --scoring errors   --out experiments/results_holoclean
        Run --dataset $ds --backends "holoclean_keep:csvs/holoclean_keep_${ds}_mapped.csv" @A --scoring detected --out experiments/results_holoclean_keep_oracle
    }
    Run --dataset flights --backends "retclean:csvs/retclean_flights_mapped.csv" @A --scoring detected --out experiments/results_retclean

    Step "2/8  Table 3: the same logs behind a real detector (Raha) and a constraint detector"
    foreach ($ds in $SMALL) {
        Run --dataset $ds --backends "baran_raha:csvs/baran_raha_${ds}_mapped.csv" @A --scoring detected `
            --detection "log:csvs/raha_${ds}_detected.csv" --out experiments/results_raha_baran
        Run --dataset $ds --backends "jellyfish:csvs/jellyfish_${ds}_mapped.csv"   @A --scoring detected `
            --detection "log:csvs/raha_${ds}_detected.csv" --out experiments/results_raha_jelly
    }
    foreach ($ds in @("hospital", "beers", "flights")) {
        Run --dataset $ds --backends "bclean_keep:csvs/bclean_keep_${ds}_mapped.csv"       @A --scoring detected `
            --detection "log:csvs/raha_${ds}_detected.csv" --out experiments/results_raha_bclean
        Run --dataset $ds --backends "holoclean_keep:csvs/holoclean_keep_${ds}_mapped.csv" @A --scoring detected `
            --detection "log:csvs/raha_${ds}_detected.csv" --out experiments/results_raha_holoclean
    }
    foreach ($ds in @("hospital", "flights")) {
        Run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv" @A --scoring detected `
            --detection constraints --out experiments/results_constraints_baran
    }
    Run --dataset flights --backends "jellyfish:csvs/jellyfish_flights_mapped.csv" @A --scoring detected `
        --detection constraints --out experiments/results_constraints_jelly

    Step "3/8  Table 2: Theorems 1 and 2 (untrusted and trusted corruption)"
    function Poison($Dataset, $Backend, $Alpha, $Scoring, $OutDir) {
        Run --dataset $Dataset --backends $Backend @P --alphas $Alpha --scoring $Scoring                  --out "experiments/$OutDir"
        Run --dataset $Dataset --backends $Backend @P --alphas $Alpha --scoring $Scoring --trusted-poison --out "experiments/$OutDir"
    }
    # Table 2 uses the five Baran/BClean configurations; the two Jellyfish ones are run
    # because Section 7.6 reports that they cannot carry a Theorem 2 exhibit (the gate
    # certifies nothing, so no poisoned cell can reach the applied set).
    Poison beers    "jellyfish:csvs/jellyfish_beers_mapped.csv"    0.1 detected results_poison_jellyfish_beers
    Poison hospital "jellyfish:csvs/jellyfish_hospital_mapped.csv" 0.2 detected results_poison_jellyfish_hospital
    Poison hospital "baran:csvs/baran_hospital_mapped.csv"         0.2 errors   results_poison_baran_hospital
    Poison beers    "baran:csvs/baran_beers_mapped.csv"            0.2 errors   results_poison_baran_beers
    Poison rayyan   "baran:csvs/baran_rayyan_mapped.csv"           0.2 errors   results_poison_baran_rayyan
    Poison hospital "bclean:csvs/bclean_hospital_mapped.csv"       0.2 errors   results_poison_bclean_hospital
    Poison flights  "bclean:csvs/bclean_flights_mapped.csv"        0.2 errors   results_poison_bclean_flights

    Step "4/8  Baran reruns: five extra draws per dataset (Table 1 medians, Section 9)"
    foreach ($ds in $SMALL) {
        foreach ($i in 1..5) {
            Run --dataset $ds --backends "baran:csvs/baran_var_${ds}/$i/baran_${ds}_mapped.csv" `
                @A --scoring errors --out "experiments/results_baran_var_${ds}/$i"
        }
    }

    if (-not $Quick) {
        Step "5/8  tax (200K rows, 3M cells): oracle, Raha, Jellyfish shard, six Baran draws"
        Run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" @A --scoring errors   --out experiments/results_baran_errors
        Run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" @A --scoring detected --out experiments/results_baran_detected
        Run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" @A --scoring detected `
            --detection "log:csvs/raha_tax_detected.csv" --out experiments/results_raha_baran
        # --row-range is mandatory: the Jellyfish tax log covers rows [0,25000) only.
        Run --dataset tax --backends "jellyfish:csvs/jellyfish_tax_mapped.csv" @A --scoring detected `
            --row-range 0 25000 --out experiments/results_jelly_tax25k
        foreach ($i in 1..5) {
            Run --dataset tax --backends "baran:csvs/baran_var_tax/$i/baran_tax_mapped.csv" `
                @A --scoring errors --out "experiments/results_baran_var_tax/$i"
        }
    } else {
        Write-Host "  [-Quick] skipping tax; its Table 1 / Table 3 rows and Section 7.3-7.5 numbers will be stale"
    }

    Step "6/8  audits"
    Invoke-Checked $PY audit_proposers.py            # accuracy on error cells, pi_0, harm            -> proposer_accuracy.csv
    Invoke-Checked $PY audit_baran_median.py         # Table 1's Baran rows as medians over draws     -> baran_median.csv
    # audit_alpha_ceiling.py expands its own globs, so the pattern is passed through quoted,
    # exactly as reproduce.sh does.
    Invoke-Checked $PY audit_alpha_ceiling.py --logs "csvs/baran_tax_mapped.csv" "csvs/baran_var_tax/*/baran_tax_mapped.csv" `
        --per-stratum --out experiments/tax_draws.csv   # per-draw, per-column (Sections 7.2-7.4)
    Invoke-Checked $PY audit_calibration.py          # is confidence a probability?                  -> calibration.csv
    Invoke-Checked $PY audit_recalibration.py        # does Platt/isotonic fix it?                    -> recalibration.csv
    Invoke-Checked $PY audit_heldout_budget.py       # the naive held-out threshold vs the bound      -> heldout_budget.csv
    Invoke-Checked $PY audit_sensitivity.py          # |Lambda| x delta grid                          -> sensitivity.csv
    Invoke-Checked $PY audit_label_budget.py         # what certification costs in labels             -> label_budget.csv
    Invoke-Checked $PY audit_drift_probe.py          # source-split drift probe (Section 9)           -> drift_probe.csv
    # error drop rate, apply-all vs CARE -> error_drop.csv (+ tab5)
    if ($Quick) {
        Invoke-Checked $PY audit_error_drop.py --raha --skip-tax --tex paper/tables/tab5_error_drop.tex
    } else {
        Invoke-Checked $PY audit_error_drop.py --raha --tex paper/tables/tab5_error_drop.tex
    }
    # Controlled error-rate sweep. The variant logs ship in csvs/*_ni_*; the ceilings were
    # sealed before calibration and audit_sweep.py checks the seal. The join reads the stored
    # per-variant sweeps in experiments/results_sweep_baran_var/ (regenerating those from the
    # variant logs is the loop in docs/REPRODUCE.md section 3).
    if (Get-ChildItem -Path "csvs" -Filter "*_ni_*_mapped.csv" -ErrorAction SilentlyContinue) {
        & $PY audit_sweep.py
        if ($LASTEXITCODE -ne 0) { Write-Host "  (sweep join incomplete; see docs/REPRODUCE.md section 3)" }
    } else {
        Write-Host "  [skip] no csvs/*_ni_*_mapped.csv -- see docs/REPRODUCE.md section 3"
    }

    Step "7/8  governance cost per repair"
    Invoke-Checked $PY -m bench.throughput --datasets flights hospital beers --exact-sample 50 --out experiments/throughput.csv

    Step "8/8  figures and tables"
    Invoke-Checked $PY make_figures.py
}

Step "verification"
Write-Host "-- the fast verification path is exact, not approximate"
Invoke-Checked $PY tools/verify_fastpath_equivalence.py
Write-Host "-- its precondition is checked at run time and its failure is handled"
Invoke-Checked $PY tools/verify_precondition_fallback.py
Write-Host "-- every stored sweep regenerates from the documented command"
if ($Quick) {
    Invoke-Checked $PY tools/verify_results_reproduce.py --only hospital beers flights rayyan
} else {
    Invoke-Checked $PY tools/verify_results_reproduce.py
}

Write-Host ""
Write-Host "done. figures in paper/figures, tables in paper/tables" -ForegroundColor White
