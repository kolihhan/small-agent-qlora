$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RunRoot = Join-Path $Repo 'runs\p4-base-diagnostic-v1'
$Preflight = 'C:\Personal_Projects\ai-projects\.runtimes\p4-canonical-preflight-20260829-a.json'
$Python = 'C:\Personal_Projects\ai-projects\.runtimes\small-agent-qlora-p4-model-py311-v3\Scripts\python.exe'
$Model = 'C:\Users\lihha\.cache\huggingface\hub\models--Qwen--Qwen3.5-4B\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
$Dataset = 'C:\Users\lihha\.cache\huggingface\hub\datasets--gaia-benchmark--GAIA\snapshots\682dd723ee1e1697e00360edccf2366dc8418dd9'
if (-not (Test-Path -LiteralPath $Preflight)) { throw 'Frozen consolidated preflight is missing.' }
$PreflightData = Get-Content -LiteralPath $Preflight -Raw | ConvertFrom-Json
if ($PreflightData.status -ne 'PASS' -or $PreflightData.canonical_output_absent -ne $true) { throw 'Frozen consolidated preflight did not pass.' }
if (Test-Path -LiteralPath $RunRoot) { throw "Refusing to overwrite canonical output: $RunRoot" }
New-Item -ItemType Directory -Path $RunRoot | Out-Null
$Stdout = Join-Path $RunRoot 'stdout.log'
$Stderr = Join-Path $RunRoot 'stderr.log'
$Resource = Join-Path $RunRoot 'resource-monitor.json'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$Arguments = @(
    '-B', '-m', 'gaia_small_agent', 'gaia-eval',
    '--partition', 'diagnostic', '--backend', 'transformers',
    '--source', $Dataset, '--hf-model', $Model,
    '--work-root', $RunRoot, '--manifest', (Join-Path $RunRoot 'manifest.json'),
    '--protected-questions', (Join-Path $RunRoot 'protected-question-hashes.json'),
    '--seed', 'gaia-local-v2', '--gaia-revision', '682dd723ee1e1697e00360edccf2366dc8418dd9',
    '--max-steps', '12', '--max-new-tokens', '512',
    '--cache-implementation', 'default'
)
$Started = Get-Date
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Samples = @()
$LowRamSince = $null
$GuardReason = $null
while (-not $Process.HasExited) {
    $OS = Get-CimInstance Win32_OperatingSystem
    $Available = [int64]$OS.FreePhysicalMemory * 1024
    $GpuParts = ((& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits) -split ',')
    $Samples += [pscustomobject]@{
        elapsed_s = [math]::Round(((Get-Date) - $Started).TotalSeconds, 3)
        available_ram_bytes = $Available
        gpu_used_mib = [int]$GpuParts[0].Trim()
        gpu_free_mib = [int]$GpuParts[1].Trim()
    }
    if ($Available -lt 536870912) {
        if ($null -eq $LowRamSince) { $LowRamSince = Get-Date }
        elseif (((Get-Date) - $LowRamSince).TotalSeconds -gt 5) { $GuardReason = 'available_ram_below_512MiB_for_more_than_5_seconds' }
    } else { $LowRamSince = $null }
    if (((Get-Date) - $Started).TotalHours -gt 4) { $GuardReason = 'wall_timeout_4h' }
    if ($null -ne $GuardReason) { Stop-Process -Id $Process.Id -Force; break }
    Start-Sleep -Seconds 1
    $Process.Refresh()
}
$Process.WaitForExit()
$Finished = Get-Date
$Payload = [ordered]@{
    schema_version = 'p4-resource-monitor/v1'
    started_at = $Started.ToUniversalTime().ToString('o')
    finished_at = $Finished.ToUniversalTime().ToString('o')
    wall_time_s = [math]::Round(($Finished - $Started).TotalSeconds, 3)
    process_exit_code = $Process.ExitCode
    guard_reason = $GuardReason
    sample_count = $Samples.Count
    min_available_ram_bytes = ($Samples.available_ram_bytes | Measure-Object -Minimum).Minimum
    peak_gpu_used_mib = ($Samples.gpu_used_mib | Measure-Object -Maximum).Maximum
    min_gpu_free_mib = ($Samples.gpu_free_mib | Measure-Object -Minimum).Minimum
    samples = $Samples
}
$Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Resource -Encoding utf8
$Payload | Select-Object wall_time_s,process_exit_code,guard_reason,sample_count,min_available_ram_bytes,peak_gpu_used_mib,min_gpu_free_mib | ConvertTo-Json -Compress
if ($Process.ExitCode -ne 0 -or $null -ne $GuardReason) { exit 1 }
