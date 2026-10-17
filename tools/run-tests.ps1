<#
.SYNOPSIS
    Run the pfc-models test suite in both PFC runtimes.

.DESCRIPTION
    Every model test gates on the backend the active runtime can import, so the
    suite has to be run once per venv to cover the catalogue: .venv-pfc-fftw
    covers logpfc_cpu and stdpfc_cpu, .venv-pfc-cupy covers logpfc, logpfc_extv,
    the loghydro family and stdpfc. Tests for the backend that is absent report
    as skipped, not as passes - read the skip count, not just the exit code.

    Collects tests/ (substrate contract plus one smoke test per model) and
    src/pfc_models/models/ (each model's test_regression.py, which also carries
    that model's physics checks).

    Windows PowerShell 5.1 compatible.

.EXAMPLE
    .\tools\run-tests.ps1
    Both runtimes, everything.

.EXAMPLE
    .\tools\run-tests.ps1 -Filter stdpfc
    Both runtimes, only tests whose name matches (pytest -k).

.EXAMPLE
    .\tools\run-tests.ps1 -Filter relaxes
    The smoke tests alone - one per model, each asserting free-energy relaxation.

.EXAMPLE
    .\tools\run-tests.ps1 -RecordGpuReference
    Print the stdpfc (CuPy) regression reference for pasting into
    src/pfc_models/models/stdpfc/test_regression.py. Runs nothing else.
#>
[CmdletBinding()]
param(
    [string] $RepoRoot = '',
    [string] $CpuVenv = 'Z:\research\.venv-pfc-fftw',
    [string] $GpuVenv = 'Z:\research\.venv-pfc-cupy',
    [string] $Filter = '',
    [switch] $SkipCpu,
    [switch] $SkipGpu,
    [switch] $RecordGpuReference
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is not populated in the param block under every 5.1 host, so the
# repo root is resolved here instead of as a parameter default.
if ([string]::IsNullOrEmpty($RepoRoot)) {
    $scriptDir = $PSScriptRoot
    if ([string]::IsNullOrEmpty($scriptDir)) {
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    $RepoRoot = Split-Path -Parent $scriptDir
}

function Get-Interpreter {
    param([string] $VenvPath, [string] $Label)
    $exe = Join-Path $VenvPath 'Scripts\python.exe'
    if (-not (Test-Path $exe)) {
        Write-Host ("  {0}: interpreter not found at {1} - skipping" -f $Label, $exe) -ForegroundColor Yellow
        return $null
    }
    return $exe
}

function Invoke-Suite {
    param([string] $Python, [string] $Label, [string[]] $Targets, [string] $Match)

    Write-Host ''
    Write-Host ("=== {0}" -f $Label) -ForegroundColor Cyan

    # Out-Host keeps native stdout out of this function's output stream; without it
    # the pytest transcript is captured into the return value alongside the exit code.
    & $Python -c "import sys; print('   ', sys.version.split()[0], sys.executable)" | Out-Host

    $pytestArgs = @('-m', 'pytest') + $Targets + @('-v', '--no-header', '-p', 'no:cacheprovider')
    if (-not [string]::IsNullOrEmpty($Match)) {
        $pytestArgs = $pytestArgs + @('-k', $Match)
    }
    & $Python $pytestArgs | Out-Host
    $code = $LASTEXITCODE

    if ($code -eq 0) {
        Write-Host ("  {0}: OK" -f $Label) -ForegroundColor Green
    } else {
        Write-Host ("  {0}: FAILED (pytest exit {1})" -f $Label, $code) -ForegroundColor Red
    }
    return $code
}

Push-Location $RepoRoot
try {
    if ($RecordGpuReference) {
        $gpu = Get-Interpreter -VenvPath $GpuVenv -Label 'GPU runtime'
        if ($null -eq $gpu) { exit 1 }
        Write-Host 'Recording the stdpfc (CuPy) regression reference...' -ForegroundColor Cyan
        & $gpu -m pfc_models.models.stdpfc.test_regression
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            Write-Host "  recorder failed (exit $code)" -ForegroundColor Red
        } else {
            Write-Host '  Paste the printed value into src/pfc_models/models/stdpfc/test_regression.py' -ForegroundColor Green
            Write-Host '  and delete the @pytest.mark.skip above test_free_energy_at_fixed_step.' -ForegroundColor Green
        }
        exit $code
    }

    $targets = @('tests', 'src/pfc_models/models')
    $failures = 0
    $ran = 0

    if (-not $SkipCpu) {
        $cpu = Get-Interpreter -VenvPath $CpuVenv -Label 'CPU runtime (.venv-pfc-fftw)'
        if ($null -ne $cpu) {
            $ran = $ran + 1
            $code = Invoke-Suite -Python $cpu -Label 'CPU runtime (.venv-pfc-fftw)' -Targets $targets -Match $Filter
            if ($code -ne 0) { $failures = $failures + 1 }
        }
    }

    if (-not $SkipGpu) {
        $gpu = Get-Interpreter -VenvPath $GpuVenv -Label 'GPU runtime (.venv-pfc-cupy)'
        if ($null -ne $gpu) {
            $ran = $ran + 1
            $code = Invoke-Suite -Python $gpu -Label 'GPU runtime (.venv-pfc-cupy)' -Targets $targets -Match $Filter
            if ($code -ne 0) { $failures = $failures + 1 }
        }
    }

    Write-Host ''
    if ($ran -eq 0) {
        Write-Host 'No runtime was available; nothing ran.' -ForegroundColor Red
        exit 1
    }
    if ($failures -eq 0) {
        Write-Host ("All {0} runtime(s) green." -f $ran) -ForegroundColor Green
        exit 0
    }
    Write-Host ("{0} of {1} runtime(s) failed." -f $failures, $ran) -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
