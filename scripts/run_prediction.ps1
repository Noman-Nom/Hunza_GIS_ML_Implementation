# run_prediction.ps1 — Apply trained model to all rasters
# =========================================================

$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $script_dir

if (-not (Test-Path .venv\Scripts\Activate.ps1)) {
    Write-Host "ERROR: .venv not found" -ForegroundColor Red
    exit 1
}

# Activate venv if not already active
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating .venv..." -ForegroundColor Cyan
    & .venv\Scripts\Activate.ps1
}

try {
    Write-Host "`n" + ("=" * 70) -ForegroundColor Green
    Write-Host "PREDICTING LAND COVER FOR ALL YEARS (2020-2023)" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Green
    Write-Host "This will apply the trained RF model to raw rasters...`n" -ForegroundColor Cyan

    python predict_all_years.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: predict_all_years.py failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "`n" + ("=" * 70) -ForegroundColor Green
    Write-Host "SUCCESS: Predictions complete!" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Green
    Write-Host "Check outputs directory: ..\outputs\" -ForegroundColor Cyan

} finally {
    Pop-Location
}
