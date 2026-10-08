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

# generate-policy-tasks — deterministic task definitions, non-GAIA.
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
foreach ($requiredTool in @('read','inspect','python','search')) {
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
        training = '64 deterministic independent non-GAIA policy tasks; >=32 policy-verified successes required with read/inspect/python/search coverage'
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
        comparison = $comparison
    }
    resources = $resources
    artifacts = $artifacts
}
$summary | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $SummaryPath -Encoding utf8
}
catch {
    try { Write-IncompleteSummary $_.Exception.Message } catch {}
    throw
}
