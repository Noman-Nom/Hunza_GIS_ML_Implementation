# run_training.ps1 — Execute diagnostic and training scripts
# ============================================================
# Usage: .\run_training.ps1

# Activate venv if not already active
if (-not (Test-Path .venv\Scripts\Activate.ps1)) {
    Write-Host "ERROR: .venv not found. Run from project root." -ForegroundColor Red
    exit 1
}

# Determine if we're in the venv
$venv_prompt = $env:VIRTUAL_ENV
if (-not $venv_prompt) {
    Write-Host "[*] Activating .venv..." -ForegroundColor Cyan
    & .venv\Scripts\Activate.ps1
}

# Get to scripts directory
$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $script_dir

try {
    # Step 1: Diagnose leakage
    Write-Host "`n" + ("=" * 70) -ForegroundColor Green
    Write-Host "STEP 1: DIAGNOSE LEAKAGE & SEPARABILITY" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Green
    Write-Host "This will analyze why you're getting 100% accuracy..." -ForegroundColor Cyan

    python diagnose_leakage.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: diagnose_leakage.py failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "`nDiagnostic complete. Check figures directory for visualizations." -ForegroundColor Cyan
    Write-Host "Press Enter to continue with training..." -ForegroundColor Yellow
    Read-Host

    # Step 2: Train model
    Write-Host "`n" + ("=" * 70) -ForegroundColor Green
    Write-Host "STEP 2: TRAIN RANDOM FOREST" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Green

    python train_random_forest_v2.py --mode distillation
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: train_random_forest_v2.py failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "`n" + ("=" * 70) -ForegroundColor Green
    Write-Host "SUCCESS: All training steps complete!" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Green
    Write-Host "Check these directories:" -ForegroundColor Cyan
    Write-Host "  Figures:  ..\figures\" -ForegroundColor Cyan
    Write-Host "  Models:   ..\models\" -ForegroundColor Cyan

} finally {
    Pop-Location
}
