<#
    Turns Srish off: unloads the models from memory and stops the API and the
    portal. The app's badge flips to Offline within about 20 seconds, because
    /health starts reporting ready:false.

        powershell -ExecutionPolicy Bypass -File scripts\stop_srish.ps1

    By default the Ollama service itself is left running, since other things
    may be using it. -All stops that too.
#>

[CmdletBinding()]
param(
    [int]    $ApiPort    = 8000,
    [int]    $PortalPort = 8501,
    [switch] $All,          # also stop the Ollama service
    [switch] $KeepServers,  # only unload the models, leave API and portal up
    [switch] $Quiet
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot

function Say($msg, $colour = "Gray") {
    if (-not $Quiet) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) -ForegroundColor $colour }
}

Say "=== Srish shutdown ===" "Cyan"

# ── Unload the models ────────────────────────────────────────────────────────
# keep_alive 0 tells Ollama to drop the model as soon as this call returns,
# which frees the VRAM immediately instead of waiting out the timer.
$router = if ($env:SRISH_ROUTER_MODEL) { $env:SRISH_ROUTER_MODEL } else { "qwen2.5:7b" }
$chat   = if ($env:SRISH_CHAT_MODEL)   { $env:SRISH_CHAT_MODEL }   else { "qwen2.5:7b" }

foreach ($model in ($router, $chat | Select-Object -Unique)) {
    try {
        $body = @{ model = $model; prompt = ""; keep_alive = 0 } | ConvertTo-Json
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post `
            -Body $body -ContentType "application/json" -TimeoutSec 30 | Out-Null
        Say "Unloaded $model" "Green"
    } catch {
        Say "Could not unload $model (Ollama may already be stopped)" "Yellow"
    }
}

# ── Stop our own servers ─────────────────────────────────────────────────────
if (-not $KeepServers) {
    foreach ($pair in @(@("api_server", $ApiPort), @("admin_portal", $PortalPort))) {
        $name, $port = $pair
        $procs = Get-CimInstance Win32_Process |
                 Where-Object { $_.CommandLine -like "*$name*" -and $_.CommandLine -like "*$port*" }
        if ($procs) {
            $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Say "Stopped $name on $port" "Green"
        } else {
            Say "$name was not running on $port"
        }
    }
}

# ── Optionally stop Ollama ───────────────────────────────────────────────────
if ($All) {
    $o = Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue
    if ($o) {
        $o | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
        Say "Stopped the Ollama service" "Green"
    } else {
        Say "Ollama was not running"
    }
}

Say ""
Say "=== off ===" "Cyan"
Say "The app will show Offline within ~20s (it re-checks /health on a timer)."
Say "Bring it back with:  srish on"
