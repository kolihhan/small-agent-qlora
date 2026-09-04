param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PortfolioRoot = Split-Path -Parent $Repo
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
$ModelRevision = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
$GaiaRevision = '682dd723ee1e1697e00360edccf2366dc8418dd9'
$Seed = 'gaia-local-v2'
$MinVerifiedTrajectories = 32

$SummaryPath = Join-Path $RunRoot 'pipeline-summary.json'
$ProtectedQuestions = Join-Path $RunRoot 'protected-question-hashes.json'
$DiagnosticRoot = Join-Path $RunRoot 'diagnostic25'
$PolicyTasks = Join-Path $RunRoot 'policy-tasks.jsonl'
$TrajectoryRoot = Join-Path $RunRoot 'trajectory-collection'
$Trajectories = Join-Path $RunRoot 'verified-trajectories.jsonl'
$AdapterRoot = Join-Path $RunRoot 'adapter'
$BaseEvalRoot = Join-Path $RunRoot 'evaluation-base'
$LoraEvalRoot = Join-Path $RunRoot 'evaluation-lora'
$ComparisonPath = Join-Path $RunRoot 'comparison.json'
$ResourceRoot = Join-Path $RunRoot 'resources'
$LogRoot = Join-Path $RunRoot 'logs'

New-Item -ItemType Directory -Force -Path $RunRoot, $ResourceRoot, $LogRoot | Out-Null

function Read-Json([string]$Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-Sha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Archive-ExistingAttemptFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $archived = "$Path.attempt-$stamp"
    Move-Item -LiteralPath $Path -Destination $archived
    return $archived
}

function Write-IncompleteSummary([string]$Message) {
    $resources = [ordered]@{}
    foreach ($name in @('diagnostic25','collect-trajectories','train-qlora','evaluation-base','evaluation-lora')) {
        $path = Join-Path $ResourceRoot "$name.json"
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            try { $resources[$name] = Read-Json $path } catch {}
        }
    }
    $candidates = [ordered]@{
        protected_questions = $ProtectedQuestions
        diagnostic_manifest = (Join-Path $DiagnosticRoot 'manifest.json')
        diagnostic_results = (Join-Path $DiagnosticRoot 'results.jsonl')
        diagnostic_summary = (Join-Path $DiagnosticRoot 'summary.json')
        policy_tasks = $PolicyTasks
        verified_trajectories = $Trajectories
        trajectory_summary = "$Trajectories.summary.json"
        adapter_config = (Join-Path $AdapterRoot 'adapter_config.json')
        base_summary = (Join-Path $BaseEvalRoot 'summary.json')
        lora_summary = (Join-Path $LoraEvalRoot 'summary.json')
        comparison = $ComparisonPath
    }
    $available = [ordered]@{}
    foreach ($name in $candidates.Keys) {
        $path = $candidates[$name]
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $available[$name] = [ordered]@{ path = $path; sha256 = Get-Sha256 $path }
        }
    }
    $payload = [ordered]@{
        schema_version = 'p4-full-pipeline/v2'
        status = 'incomplete'
        generated_at = (Get-Date).ToUniversalTime().ToString('o')
        error = $Message
        protocol = [ordered]@{
            model = 'Qwen/Qwen3.5-4B'
            model_revision = $ModelRevision
            gaia_revision = $GaiaRevision
            seed = $Seed
        }
        resources = $resources
        available_artifacts = $available
    }
    $payload | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $SummaryPath -Encoding utf8
}

function Resolve-P4Python {
    if ($env:P4_PYTHON -and (Test-Path -LiteralPath $env:P4_PYTHON -PathType Leaf)) {
        return (Resolve-Path -LiteralPath $env:P4_PYTHON).Path
    }
    $candidates = @(
        (Join-Path $PortfolioRoot '.runtimes\small-agent-qlora-p4-model-py311-v3\Scripts\python.exe'),
        (Join-Path $Repo '.venv\Scripts\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return (Resolve-Path -LiteralPath $candidate).Path }
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw 'No P4 Python interpreter found. Set P4_PYTHON to the Python environment containing the P4 runtime/train dependencies.'
}

function Resolve-FrozenModel {
    if ($env:P4_HF_MODEL) {
        if (-not (Test-Path -LiteralPath $env:P4_HF_MODEL -PathType Container)) {
            throw 'P4_HF_MODEL must point to the frozen local Qwen3.5-4B snapshot directory.'
        }
        return (Resolve-Path -LiteralPath $env:P4_HF_MODEL).Path
    }
    $candidate = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen3.5-4B\snapshots\$ModelRevision"
    if (Test-Path -LiteralPath $candidate -PathType Container) { return (Resolve-Path -LiteralPath $candidate).Path }
    throw "Frozen Qwen3.5-4B snapshot $ModelRevision not found. Set P4_HF_MODEL to that exact local snapshot."
}

function Resolve-FrozenGaia {
    if ($env:P4_GAIA_SOURCE) {
        if (-not (Test-Path -LiteralPath $env:P4_GAIA_SOURCE -PathType Container)) {
            throw 'P4_GAIA_SOURCE must point to the frozen local GAIA dataset snapshot directory.'
        }
        return (Resolve-Path -LiteralPath $env:P4_GAIA_SOURCE).Path
    }
    $candidate = Join-Path $env:USERPROFILE ".cache\huggingface\hub\datasets--gaia-benchmark--GAIA\snapshots\$GaiaRevision"
    if (Test-Path -LiteralPath $candidate -PathType Container) { return (Resolve-Path -LiteralPath $candidate).Path }
    throw "Frozen GAIA snapshot $GaiaRevision not found. Set P4_GAIA_SOURCE to that exact local snapshot."
}

function Assert-SummaryTotal([string]$Path, [int]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $payload = Read-Json $Path
        return [int]$payload.total -eq $Expected
    } catch { return $false }
}

function Invoke-MonitoredPython {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [double]$MaxHours = 12
    )

    $stdout = Join-Path $LogRoot "$Name.stdout.log"
    $stderr = Join-Path $LogRoot "$Name.stderr.log"
    $resource = Join-Path $ResourceRoot "$Name.json"
    Archive-ExistingAttemptFile $stdout | Out-Null
    Archive-ExistingAttemptFile $stderr | Out-Null
    Archive-ExistingAttemptFile $resource | Out-Null
    $started = Get-Date
    $samples = [System.Collections.Generic.List[object]]::new()
    $lowRamSince = $null
    $guardReason = $null

    $process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

    while (-not $process.HasExited) {
        $available = $null
        try {
            $os = Get-CimInstance Win32_OperatingSystem
            $available = [int64]$os.FreePhysicalMemory * 1024
        } catch {}

        $gpuUsed = $null
        $gpuFree = $null
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            try {
                $parts = ((& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | Select-Object -First 1) -split ',')
                $gpuUsed = [int]$parts[0].Trim()
                $gpuFree = [int]$parts[1].Trim()
            } catch {}
        }

        $samples.Add([pscustomobject]@{
            elapsed_s = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
            available_ram_bytes = $available
            gpu_used_mib = $gpuUsed
            gpu_free_mib = $gpuFree
        })

        if ($null -ne $available -and $available -lt 536870912) {
            if ($null -eq $lowRamSince) { $lowRamSince = Get-Date }
            elseif (((Get-Date) - $lowRamSince).TotalSeconds -gt 5) {
                $guardReason = 'available_ram_below_512MiB_for_more_than_5_seconds'
            }
        } else {
            $lowRamSince = $null
        }
        if (((Get-Date) - $started).TotalHours -gt $MaxHours) {
            $guardReason = "wall_timeout_${MaxHours}h"
        }
        if ($guardReason) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            break
        }
        Start-Sleep -Seconds 2
        $process.Refresh()
    }

    $process.WaitForExit()
    $finished = Get-Date
    $ramValues = @($samples | Where-Object { $null -ne $_.available_ram_bytes } | ForEach-Object available_ram_bytes)
    $gpuUsedValues = @($samples | Where-Object { $null -ne $_.gpu_used_mib } | ForEach-Object gpu_used_mib)
    $gpuFreeValues = @($samples | Where-Object { $null -ne $_.gpu_free_mib } | ForEach-Object gpu_free_mib)
    $payload = [ordered]@{
        schema_version = 'p4-resource-monitor/v2'
        stage = $Name
        started_at = $started.ToUniversalTime().ToString('o')
        finished_at = $finished.ToUniversalTime().ToString('o')
        wall_time_s = [math]::Round(($finished - $started).TotalSeconds, 3)
        process_exit_code = $process.ExitCode
        guard_reason = $guardReason
        sample_count = $samples.Count
        min_available_ram_bytes = if ($ramValues.Count) { ($ramValues | Measure-Object -Minimum).Minimum } else { $null }
        peak_gpu_used_mib = if ($gpuUsedValues.Count) { ($gpuUsedValues | Measure-Object -Maximum).Maximum } else { $null }
        min_gpu_free_mib = if ($gpuFreeValues.Count) { ($gpuFreeValues | Measure-Object -Minimum).Minimum } else { $null }
        stdout = $stdout
        stderr = $stderr
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resource -Encoding utf8

    if ($guardReason) { throw "$Name stopped by resource guard: $guardReason. Partial artifacts were preserved." }
    if ($process.ExitCode -ne 0) { throw "$Name failed with exit code $($process.ExitCode). See $stderr" }
}

function Invoke-PlainPython([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

function Stop-StaleOllamaModelRunners {
    # Transformers stages must own the GPU/RAM runtime; remove only Ollama's
    # model runner processes, leaving the Ollama service itself available.
    $runners = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq 'llama-server.exe' -and $_.CommandLine -like '*\.ollama\models\*' })
    foreach ($runner in $runners) {
        Stop-Process -Id ([int]$runner.ProcessId) -Force -ErrorAction SilentlyContinue
    }
}

try {
$Python = Resolve-P4Python
$Model = Resolve-FrozenModel
$Dataset = Resolve-FrozenGaia
Stop-StaleOllamaModelRunners

# Freeze local-only identities. The exact snapshot directory names are the revisions used by this experiment.
if ((Split-Path -Leaf $Model) -ne $ModelRevision) {
    throw "P4 model snapshot must be exact revision $ModelRevision; got $Model"
}
if ((Split-Path -Leaf $Dataset) -ne $GaiaRevision) {
    throw "P4 GAIA snapshot must be exact revision $GaiaRevision; got $Dataset"
}

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Join-Path $Repo 'src')

# Completed runs are immutable. Incomplete summaries are archived and the same
# RunRoot resumes from stage artifacts rather than restarting valid work.
if (Test-Path -LiteralPath $SummaryPath -PathType Leaf) {
    $existing = Read-Json $SummaryPath
    if ($existing.status -eq 'complete') {
        Write-Output "P4 pipeline already complete: $SummaryPath"
        exit 0
    }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $archived = Join-Path $RunRoot "pipeline-summary.incomplete-$stamp.json"
    Move-Item -LiteralPath $SummaryPath -Destination $archived
    Write-Output "Resuming incomplete P4 pipeline; previous summary preserved at $archived"
}

# Stage 0: cheap verification. No model inference.
Push-Location $Repo
try {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) { throw 'uv is required for the project test gate.' }
    $pytestBaseTemp = Join-Path $RunRoot ".pytest-temp"
    if (Test-Path -LiteralPath $pytestBaseTemp) {
        Remove-Item -LiteralPath $pytestBaseTemp -Recurse -Force -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Force -Path $pytestBaseTemp | Out-Null
    & $uv.Source run --extra dev --extra files python -m pytest -q -p no:cacheprovider --basetemp $pytestBaseTemp
    if ($LASTEXITCODE -ne 0) { throw 'P4 test gate failed.' }
    & $Python -B -m compileall -q src tests
    if ($LASTEXITCODE -ne 0) { throw 'P4 compile gate failed.' }
} finally {
    Pop-Location
}

# diagnostic25 — development only. Resumable; it also freezes protected-question-hashes.json.
$diagnosticSummary = Join-Path $DiagnosticRoot 'summary.json'
if (-not (Assert-SummaryTotal $diagnosticSummary 25)) {
    Invoke-MonitoredPython -Name 'diagnostic25' -Arguments @(
        '-B','-m','gaia_small_agent','gaia-eval',
        '--partition','diagnostic','--backend','transformers',
        '--source',$Dataset,'--hf-model',$Model,
        '--work-root',$DiagnosticRoot,'--manifest',(Join-Path $DiagnosticRoot 'manifest.json'),
        '--protected-questions',$ProtectedQuestions,
        '--seed',$Seed,'--gaia-revision',$GaiaRevision,
        '--max-steps','12','--max-new-tokens','512','--cache-implementation','default'
    )
}
if (-not (Assert-SummaryTotal $diagnosticSummary 25)) { throw 'diagnostic25 did not complete all 25 frozen cases.' }
if (-not (Test-Path -LiteralPath $ProtectedQuestions -PathType Leaf)) { throw 'protected-question-hashes.json was not produced.' }

# generate-policy-tasks — deterministic, non-GAIA, no web-dependent tasks.
Invoke-PlainPython @('-B','-m','gaia_small_agent','generate-policy-tasks','--output',$PolicyTasks,'--count','64','--seed','p4-policy-v1')

# collect-trajectories — restart-safe by task id; only clean verified successes are saved.
$trajectorySummary = "$Trajectories.summary.json"
if (-not (Test-Path -LiteralPath $trajectorySummary -PathType Leaf) -or [int](Read-Json $trajectorySummary).total -ne 64) {
    Invoke-MonitoredPython -Name 'collect-trajectories' -Arguments @(
        '-B','-m','gaia_small_agent','collect-trajectories',
        '--tasks',$PolicyTasks,'--output',$Trajectories,'--work-root',$TrajectoryRoot,
        '--protected-questions',$ProtectedQuestions,
        '--backend','transformers','--hf-model',$Model,
        '--max-steps','12','--max-new-tokens','512','--cache-implementation','default'
    )
}
$trajectoryStats = Read-Json $trajectorySummary
if ([int]$trajectoryStats.total -ne 64) { throw 'trajectory collection did not account for all 64 frozen policy tasks.' }
if ([int]$trajectoryStats.verified -lt $MinVerifiedTrajectories) {
    throw "Only $($trajectoryStats.verified)/64 policy-verified trajectories passed; frozen minimum is $MinVerifiedTrajectories, so QLoRA will not start."
}
$verifiedRows = @(Get-Content -LiteralPath $Trajectories | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
$coveredTools = @($verifiedRows | ForEach-Object { @($_.required_tools) } | Sort-Object -Unique)
foreach ($requiredTool in @('read','inspect','python')) {
    if ($requiredTool -notin $coveredTools) { throw "Verified trajectory set has no clean $requiredTool policy example; QLoRA will not start." }
}

# train-qlora — one frozen training treatment. Preserve any interrupted adapter directory before retrying.
$adapterConfig = Join-Path $AdapterRoot 'adapter_config.json'
$adapterModel = $null
if (Test-Path -LiteralPath (Join-Path $AdapterRoot 'adapter_model.safetensors')) { $adapterModel = Join-Path $AdapterRoot 'adapter_model.safetensors' }
elseif (Test-Path -LiteralPath (Join-Path $AdapterRoot 'adapter_model.bin')) { $adapterModel = Join-Path $AdapterRoot 'adapter_model.bin' }

if (-not $adapterModel -or -not (Test-Path -LiteralPath $adapterConfig -PathType Leaf)) {
    if (Test-Path -LiteralPath $AdapterRoot) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        Move-Item -LiteralPath $AdapterRoot -Destination "$AdapterRoot.invalid-$stamp"
    }
    Invoke-MonitoredPython -Name 'train-qlora' -Arguments @(
        '-B','-m','gaia_small_agent','train-qlora',
        '--data',$Trajectories,'--output',$AdapterRoot,
        '--hf-model',$Model,'--max-length','1536','--epochs','2.0',
        '--protected-questions',$ProtectedQuestions
    ) -MaxHours 12
    if (Test-Path -LiteralPath (Join-Path $AdapterRoot 'adapter_model.safetensors')) { $adapterModel = Join-Path $AdapterRoot 'adapter_model.safetensors' }
    elseif (Test-Path -LiteralPath (Join-Path $AdapterRoot 'adapter_model.bin')) { $adapterModel = Join-Path $AdapterRoot 'adapter_model.bin' }
}
if (-not $adapterModel -or -not (Test-Path -LiteralPath $adapterConfig -PathType Leaf)) { throw 'train-qlora completed without a loadable adapter artifact.' }

# evaluation-base — sealed evaluation100, first arm.
$baseSummaryPath = Join-Path $BaseEvalRoot 'summary.json'
if (-not (Assert-SummaryTotal $baseSummaryPath 100)) {
    Invoke-MonitoredPython -Name 'evaluation-base' -Arguments @(
        '-B','-m','gaia_small_agent','gaia-eval',
        '--partition','evaluation','--backend','transformers',
        '--source',$Dataset,'--hf-model',$Model,
        '--work-root',$BaseEvalRoot,'--manifest',(Join-Path $BaseEvalRoot 'manifest.json'),
        '--protected-questions',$ProtectedQuestions,
        '--seed',$Seed,'--gaia-revision',$GaiaRevision,
        '--max-steps','12','--max-new-tokens','512','--cache-implementation','default'
    ) -MaxHours 12
}
if (-not (Assert-SummaryTotal $baseSummaryPath 100)) { throw 'evaluation-base did not complete all 100 frozen cases.' }

# evaluation-lora — identical evaluation100 settings except adapter.
$loraSummaryPath = Join-Path $LoraEvalRoot 'summary.json'
if (-not (Assert-SummaryTotal $loraSummaryPath 100)) {
    Invoke-MonitoredPython -Name 'evaluation-lora' -Arguments @(
        '-B','-m','gaia_small_agent','gaia-eval',
        '--partition','evaluation','--backend','transformers',
        '--source',$Dataset,'--hf-model',$Model,'--adapter',$AdapterRoot,
        '--work-root',$LoraEvalRoot,'--manifest',(Join-Path $LoraEvalRoot 'manifest.json'),
        '--protected-questions',$ProtectedQuestions,
        '--seed',$Seed,'--gaia-revision',$GaiaRevision,
        '--max-steps','12','--max-new-tokens','512','--cache-implementation','default'
    ) -MaxHours 12
}
if (-not (Assert-SummaryTotal $loraSummaryPath 100)) { throw 'evaluation-lora did not complete all 100 frozen cases.' }

# compare-evals — cheap to recompute; preserve any prior attempt before overwrite.
Archive-ExistingAttemptFile $ComparisonPath | Out-Null
Invoke-PlainPython @('-B','-m','gaia_small_agent','compare-evals','--base',$BaseEvalRoot,'--adapter-run',$LoraEvalRoot,'--output',$ComparisonPath)

$diagnostic = Read-Json $diagnosticSummary
$trajectoryStats = Read-Json $trajectorySummary
$baseSummary = Read-Json $baseSummaryPath
$loraSummary = Read-Json $loraSummaryPath
$comparison = Read-Json $ComparisonPath
$adapterConfigPayload = Read-Json $adapterConfig

$artifacts = [ordered]@{}
foreach ($entry in @(
    @('protected_questions',$ProtectedQuestions),
    @('diagnostic_manifest',(Join-Path $DiagnosticRoot 'manifest.json')),
    @('diagnostic_results',(Join-Path $DiagnosticRoot 'results.jsonl')),
    @('diagnostic_summary',$diagnosticSummary),
    @('policy_tasks',$PolicyTasks),
    @('verified_trajectories',$Trajectories),
    @('trajectory_summary',$trajectorySummary),
    @('adapter_model',$adapterModel),
    @('adapter_config',$adapterConfig),
    @('base_manifest',(Join-Path $BaseEvalRoot 'manifest.json')),
    @('base_results',(Join-Path $BaseEvalRoot 'results.jsonl')),
    @('base_summary',$baseSummaryPath),
    @('lora_manifest',(Join-Path $LoraEvalRoot 'manifest.json')),
    @('lora_results',(Join-Path $LoraEvalRoot 'results.jsonl')),
    @('lora_summary',$loraSummaryPath),
    @('comparison',$ComparisonPath)
)) {
    $name = $entry[0]
    $path = $entry[1]
    $artifacts[$name] = [ordered]@{ path = $path; sha256 = Get-Sha256 $path }
}

$resources = [ordered]@{}
foreach ($name in @('diagnostic25','collect-trajectories','train-qlora','evaluation-base','evaluation-lora')) {
    $path = Join-Path $ResourceRoot "$name.json"
    if (Test-Path -LiteralPath $path -PathType Leaf) { $resources[$name] = Read-Json $path }
}

$summary = [ordered]@{
    schema_version = 'p4-full-pipeline/v2'
    status = 'complete'
    generated_at = (Get-Date).ToUniversalTime().ToString('o')
    protocol = [ordered]@{
        model = 'Qwen/Qwen3.5-4B'
        model_revision = $ModelRevision
        model_snapshot = $Model
        gaia_revision = $GaiaRevision
        gaia_snapshot = $Dataset
        seed = $Seed
        diagnostic = 'GAIA Diagnostic25 development partition'
        training = '64 deterministic independent non-GAIA policy tasks; >=32 policy-verified successes required with read/inspect/python coverage'
        evaluation = 'sealed GAIA evaluation100; Base versus identical +LoRA treatment'
    }
    diagnostic = [ordered]@{
        summary = $diagnostic
        failure_signals = $diagnostic.failure_signals
    }
    training = [ordered]@{
        task_count = [int]$trajectoryStats.total
        verified_trajectory_count = [int]$trajectoryStats.verified
        minimum_verified_trajectories = $MinVerifiedTrajectories
        covered_required_tools = $coveredTools
        failed_task_count = [int]$trajectoryStats.failed
        rejected_gaia = [int]$trajectoryStats.rejected_gaia
        rejected_protected = $trajectoryStats.rejected_protected
        adapter_path = $AdapterRoot
        adapter_sha256 = Get-Sha256 $adapterModel
        adapter_config = $adapterConfigPayload
        contamination_guard = [ordered]@{
            protected_question_hashes = $ProtectedQuestions
            protected_question_hashes_sha256 = Get-Sha256 $ProtectedQuestions
            gaia_trajectories_forbidden = $true
            exact_normalized_question_overlap_forbidden = $true
            required_provenance = $true
            clean_policy_trace_required = $true
        }
    }
    evaluation = [ordered]@{
        base = $baseSummary
        lora = $loraSummary
        paired = $comparison
    }
    resources = $resources
    artifacts = $artifacts
}

$summary | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $SummaryPath -Encoding utf8
Write-Output "P4 pipeline complete: $SummaryPath"
} catch {
    $message = $_.Exception.Message
    Write-IncompleteSummary $message
    Write-Error "P4 pipeline incomplete: $message"
    throw
}
