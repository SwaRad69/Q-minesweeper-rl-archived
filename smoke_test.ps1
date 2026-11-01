<#
.SYNOPSIS
    Run a human demonstration smoke test before collecting production data.

.DESCRIPTION
    Creates a timestamped, isolated smoke-test dataset, launches the recorder
    with diagnostics and a generous startup timeout, and then validates the
    results.

.PARAMETER Games
    Stop automatically after this many completed games (default: 0 = manual Ctrl+C).

.PARAMETER Timeout
    Seconds to wait for the initial board to become readable (default: 30).
    Increase if the site loads slowly (ads, cookie consent, slow network).

.EXAMPLE
    .\smoke_test.ps1
    .\smoke_test.ps1 -Games 2
    .\smoke_test.ps1 -Games 3 -Timeout 45
#>

param(
    [int]$Games = 0,
    [int]$Timeout = 30
)

$ErrorActionPreference = "Stop"

# --- 1. Create a timestamped smoke directory ---
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$smokeDir = "data/smoke_$stamp"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Human Data Smoke Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Dataset:  $smokeDir"
Write-Host "  Timeout:  ${Timeout}s (initial board read)"
if ($Games -gt 0) {
    Write-Host "  Games:    $Games (auto-stop)"
} else {
    Write-Host "  Games:    unlimited (press Ctrl+C to stop)"
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --- 2. Run the recorder ---
$env:MINESWEEPER_URL = "https://minesweeper.online/start/3"

$recorderArgs = @(
    "-m", "minesweeper_rl.data.record_human_dataset",
    "--dataset", $smokeDir,
    "--diagnostics",
    "--action-timeout", $Timeout
)
if ($Games -gt 0) {
    $recorderArgs += @("--games", $Games)
}

Write-Host "[1/3] Starting recorder..." -ForegroundColor Yellow
Write-Host "  Play at least 1 game. Use OPEN, FLAG, and CHORD actions." -ForegroundColor Gray
Write-Host "  Press Ctrl+C when done." -ForegroundColor Gray
Write-Host ""

python @recorderArgs

# --- 3. Validate ---
Write-Host ""
Write-Host "[2/3] Validating smoke dataset..." -ForegroundColor Yellow
Write-Host ""
python -m minesweeper_rl.data.validate_human_dataset --dataset $smokeDir

# --- 4. Loader check ---
Write-Host ""
Write-Host "[3/3] Verifying dataset loader..." -ForegroundColor Yellow
Write-Host ""
python -c @"
from minesweeper_rl.data.human_demonstrations import HumanDemonstrationDataset
ds = HumanDemonstrationDataset('$smokeDir')
print(ds.statistics().format())
assert ds.num_actions == 1440, f'Expected 1440 actions, got {ds.num_actions}'
print()
print('Loader check PASSED')
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Smoke test complete!" -ForegroundColor Green
Write-Host "  Dataset:  $smokeDir" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "If validation passed (status=OK, invalid=0, all action types > 0),"
Write-Host "you can start production collection:" -ForegroundColor White
Write-Host ""
Write-Host '  $env:MINESWEEPER_URL = "https://minesweeper.online/start/3"' -ForegroundColor Cyan
Write-Host '  python -m minesweeper_rl.data.record_human_dataset --dataset data/human --diagnostics' -ForegroundColor Cyan
Write-Host ""
