<#
reproduce.ps1 -- Windows counterpart of reproduce.sh, step for step.

    .\reproduce.ps1            # sweeps, audits, figures, verification
    .\reproduce.ps1 -Quick     # skip tax (about 16 GB of RAM) and the error-rate sweep
    .\reproduce.ps1 -Verify    # only check that stored sweeps regenerate
#>

[CmdletBinding()]
param([switch]$Quick, [switch]$Verify, [switch]$Help)

$ErrorActionPreference = "Stop"
if ($Help) { Get-Content $MyInvocation.MyCommand.Path | Select-Object -Skip 1 -First 6; exit 0 }
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$PY = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Step([string]$Message) { Write-Host ""; Write-Host "== $Message" }
function Invoke-Checked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Cmd)
    & $Cmd[0] @($Cmd[1..($Cmd.Length - 1)])
    if ($LASTEXITCODE -ne 0) { Write-Error "command failed (exit $LASTEXITCODE): $($Cmd -join ' ')"; exit $LASTEXITCODE }
}

& $PY -c "import care" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "CARE is not importable; run: pip install -r requirements.txt; pip install -e ."
    exit 1
}

$A = @("--experiment", "pareto", "--alphas", "0.05", "0.1", "0.2", "--seeds", "10", "--fast-verify", "on")
$P = @("--experiment", "poison", "--poison-fracs", "0.0", "0.05", "0.1", "0.2", "--seeds", "10", "--fast-verify", "on")
$SMALL = @("hospital", "beers", "flights", "rayyan")
function Run { param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArgs) Invoke-Checked $PY -m bench.run_study @RunArgs }

if (-not $Verify) {
    Step "1/9  oracle detection: the six cleaners (Table 4)"
    foreach ($ds in $SMALL) {
        Run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv"   @A --scoring errors   --out experiments/results_baran_errors
        Run --dataset $ds --backends "baran:csvs/baran_${ds}_mapped.csv"   @A --scoring detected --out experiments/results_baran_detected
        Run --dataset $ds --backends "gidcl:csvs/gidcl_${ds}_mapped.csv"   @A --scoring errors   --out experiments/results_gidcl
        Run --dataset $ds --backends "zeroec:csvs/zeroec_${ds}_mapped.csv" @A --scoring errors   --out experiments/results_zeroec
    }
    foreach ($ds in @("hospital", "beers", "flights")) {
        Run --dataset $ds --backends "bclean:csvs/bclean_${ds}_mapped.csv"                 @A --scoring errors   --out experiments/results_bclean
        Run --dataset $ds --backends "bclean_keep:csvs/bclean_keep_${ds}_mapped.csv"       @A --scoring detected --out experiments/results_bclean_keep_oracle
        Run --dataset $ds --backends "bclean_plus:csvs/bclean_plus_${ds}_mapped.csv"       @A --scoring errors   --out experiments/results_bclean_plus
        Run --dataset $ds --backends "holoclean:csvs/holoclean_${ds}_mapped.csv"           @A --scoring errors   --out experiments/results_holoclean
        Run --dataset $ds --backends "holoclean_keep:csvs/holoclean_keep_${ds}_mapped.csv" @A --scoring detected --out experiments/results_holoclean_keep_oracle
    }
    foreach ($ds in @("hospital", "beers", "rayyan")) {
        Run --dataset $ds --backends "gidcl_scored:csvs/gidcl_scored_${ds}_mapped.csv"       @A --scoring errors --out experiments/results_gidcl_scored
        Run --dataset $ds --backends "gidcl_scored_s1:csvs/gidcl_scored_s1_${ds}_mapped.csv" @A --scoring errors --out experiments/results_gidcl_scored_s1
        Run --dataset $ds --backends "gidcl_scored_s2:csvs/gidcl_scored_s2_${ds}_mapped.csv" @A --scoring errors --out experiments/results_gidcl_scored_s2
    }

    Step "2/9  real detectors: Raha, GIDCL's own, a constraint detector (Table 8)"
    foreach ($ds in $SMALL) {
        Run --dataset $ds --backends "baran_raha:csvs/baran_raha_${ds}_mapped.csv" @A --scoring detected `
            --detection "log:csvs/raha_${ds}_detected.csv" --out experiments/results_raha_baran
        Run --dataset $ds --backends "gidcl:csvs/gidcl_${ds}_mapped.csv" @A --scoring detected `
            --detection "log:csvs/gidcl_${ds}_detected.csv" --out experiments/results_gidcl_owndet
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

    Step "3/9  corruption through untrusted and trusted sources (Table 7)"
    function Poison([string]$ds, [string]$backend, [string]$out) {
        Run --dataset $ds --backends $backend @P --alphas 0.2 --scoring errors --out "experiments/$out"
        Run --dataset $ds --backends $backend @P --alphas 0.2 --scoring errors --trusted-poison --out "experiments/$out"
    }
    Poison hospital "baran:csvs/baran_hospital_mapped.csv"   results_poison_baran_hospital
    Poison beers    "baran:csvs/baran_beers_mapped.csv"      results_poison_baran_beers
    Poison rayyan   "baran:csvs/baran_rayyan_mapped.csv"     results_poison_baran_rayyan
    Poison hospital "bclean:csvs/bclean_hospital_mapped.csv" results_poison_bclean_hospital
    Poison flights  "bclean:csvs/bclean_flights_mapped.csv"  results_poison_bclean_flights

    Step "4/9  Baran: five more draws per table"
    foreach ($ds in $SMALL) {
        foreach ($i in 1..5) {
            Run --dataset $ds --backends "baran:csvs/baran_var_${ds}/$i/baran_${ds}_mapped.csv" `
                @A --scoring errors --out "experiments/results_baran_var_${ds}/$i"
        }
    }

    if (-not $Quick) {
        Step "5/9  tax (200K rows, 3M cells)"
        Run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" @A --scoring errors   --out experiments/results_baran_errors
        Run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" @A --scoring detected --out experiments/results_baran_detected
        Run --dataset tax --backends "baran:csvs/baran_tax_mapped.csv" @A --scoring detected `
            --detection "log:csvs/raha_tax_detected.csv" --out experiments/results_raha_baran
        Run --dataset tax --backends "gidcl:csvs/gidcl_tax_mapped.csv" @A --scoring errors --out experiments/results_gidcl
        Run --dataset tax --backends "gidcl:csvs/gidcl_tax_mapped.csv" @A --scoring detected `
            --detection "log:csvs/gidcl_tax_detected.csv" --out experiments/results_gidcl_owndet
        foreach ($i in 1..5) {
            Run --dataset tax --backends "baran:csvs/baran_var_tax/$i/baran_tax_mapped.csv" `
                @A --scoring errors --out "experiments/results_baran_var_tax/$i"
        }

        Step "6/9  controlled error-rate sweep (Ni et al. variants)"
        $logs = @(Get-ChildItem csvs -Filter "baran_*_ni_inner*_mapped.csv") + @(Get-ChildItem csvs/baran_var_ni -Recurse -Filter "baran_*_mapped.csv")
        foreach ($f in $logs) {
            $v = $f.BaseName -replace '^baran_', '' -replace '_mapped$', ''
            $rel = Resolve-Path -Relative $f.FullName
            $d = if ($rel -like "*baran_var_ni*") { $f.Directory.Name } else { "0" }
            Run --dataset $v --backends "baran:$($rel -replace '\\', '/' -replace '^\./', '')" @A --scoring errors `
                --out "experiments/results_sweep_baran_var/$v/$d"
        }
    } else {
        Write-Host "  -Quick: tax and the error-rate sweep are skipped; their numbers keep the stored values"
    }

    Step "7/9  audits"
    foreach ($s in @("audit_proposers.py", "audit_baran_median.py", "audit_calibration.py", "audit_recalibration.py",
                     "audit_heldout_budget.py", "audit_sensitivity.py", "audit_label_budget.py",
                     "audit_drift_probe.py", "audit_gidcl_confidence.py")) { Invoke-Checked $PY $s }
    if ($Quick) {
        Invoke-Checked $PY audit_error_drop.py --raha --skip-tax --tex paper/tables/tab5_error_drop.tex
    } else {
        Invoke-Checked $PY audit_alpha_ceiling.py --logs "csvs/baran_tax_mapped.csv" "csvs/baran_var_tax/*/baran_tax_mapped.csv" `
            --per-stratum --out experiments/tax_draws.csv
        Invoke-Checked $PY audit_error_drop.py --raha --tex paper/tables/tab5_error_drop.tex
        Invoke-Checked $PY audit_sweep.py
        Invoke-Checked $PY tools/verify_incremental.py tax --sample 60
    }
    Invoke-Checked $PY tools/verify_incremental.py hospital --sample 0
    foreach ($s in @("tools/audit_planner.py", "tools/audit_partitions.py", "tools/audit_trust_noise.py",
                     "tools/audit_drift_monitor.py")) { Invoke-Checked $PY $s }
    foreach ($d in 1..3) {
        Invoke-Checked $PY tools/audit_drift_monitor.py --sources flights:10 hospital:10 hospital:70 --draw $d `
            --alphas 0.2 --out "experiments/drift_monitor_d$d.csv"
    }
    Invoke-Checked $PY -c "import csv,glob,os; ps=sorted(glob.glob('experiments/drift_monitor_d[123].csv')); rows=[r for p in ps for r in csv.DictReader(open(p))]; w=csv.DictWriter(open('experiments/drift_monitor_more.csv','w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); [os.remove(p) for p in ps]"
    if ($Quick) { Write-Host "  -Quick: Table 5 (procedures) and Table 2 (scaling) keep the stored values" }
    else {
        Invoke-Checked $PY tools/audit_procedures.py --tax
        Invoke-Checked $PY tools/audit_scaling.py
    }
    Invoke-Checked $PY tools/audit_overwrite_formats.py --all

    Step "8/9  cost per repair"
    Invoke-Checked $PY -m bench.throughput --datasets flights hospital beers --exact-sample 50 --out experiments/throughput.csv

    Step "9/9  figures and tables"
    Invoke-Checked $PY make_figures.py
}

Step "verification"
Invoke-Checked $PY tools/verify_fastpath_equivalence.py
Invoke-Checked $PY tools/verify_precondition_fallback.py
Invoke-Checked $PY tools/verify_gate_first.py
if ($Quick) {
    Invoke-Checked $PY tools/verify_results_reproduce.py --only hospital beers flights rayyan
} else {
    Invoke-Checked $PY tools/verify_results_reproduce.py
}
Write-Host ""
Write-Host "done: figures in paper/figures, tables in paper/tables"
