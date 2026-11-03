<#
    Brings Srish up so both the mobile app and the admin portal can answer
    immediately, with no "first question takes 30 seconds" penalty.

    Four steps:
      1. make sure the Ollama server is running
      2. pull the model into memory and pin it there, so the first real
         question does not pay the load cost
      3. start the HTTP API on 0.0.0.0 for the phone app
      4. start the admin portal

    Run by hand:
        powershell -ExecutionPolicy Bypass -File scripts\start_srish.ps1

    Run at every login:
        powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
#>

[CmdletBinding()]
param(
    [int]    $ApiPort    = 8000,
    [int]    $PortalPort = 8501,
    [string] $KeepAlive  = "8h",     # how long models stay pinned in memory
    [switch] $NoPortal,              # API only, skip the portal
    [switch] $Quiet                  # no console windows
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory $LogDir | Out-Null }
$Stamp = Get-Date -Format "yyyy-MM-dd"
$Log = Join-Path $LogDir "start_srish_$Stamp.log"

function Say($msg, $colour = "Gray") {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $Log -Value $line -Encoding utf8
    if (-not $Quiet) { Write-Host $line -ForegroundColor $colour }
}

function Test-Port([int]$Port) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $Port)
        $c.Close(); return $true
    } catch { return $false }
}

Say "=== Srish startup ===" "Cyan"

# ── 1. Ollama server ─────────────────────────────────────────────────────────
# The desktop "Ollama app" normally autostarts at login and serves 11434. If it
# has not come up yet, start the server directly rather than waiting on it.
if (Test-Port 11434) {
    Say "Ollama already serving on 11434" "Green"
} else {
    Say "Ollama not responding; starting it"
    $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $ollama) {
        $guess = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
        if (Test-Path $guess) { $ollama = $guess }
    }
    if (-not $ollama) {
        Say "FATAL: ollama.exe not found. Install from https://ollama.com" "Red"
        exit 1
    }
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden

    $waited = 0
    while (-not (Test-Port 11434) -and $waited -lt 60) {
        Start-Sleep -Seconds 2; $waited += 2
    }
    if (Test-Port 11434) { Say "Ollama up after ${waited}s" "Green" }
    else { Say "FATAL: Ollama did not start within 60s" "Red"; exit 1 }
}

# ── 2. Warm the models ───────────────────────────────────────────────────────
# An empty prompt with keep_alive set makes Ollama load the weights and hold
# them, without generating anything. This is what removes the cold-start wait.
$router = if ($env:SRISH_ROUTER_MODEL) { $env:SRISH_ROUTER_MODEL } else { "qwen2.5:7b" }
$chat   = if ($env:SRISH_CHAT_MODEL)   { $env:SRISH_CHAT_MODEL }   else { "qwen2.5:7b" }

foreach ($model in ($router, $chat | Select-Object -Unique)) {
    Say "Warming $model (keep_alive $KeepAlive)"
    $body = @{ model = $model; prompt = ""; keep_alive = $KeepAlive } | ConvertTo-Json
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post `
            -Body $body -ContentType "application/json" -TimeoutSec 300 | Out-Null
        Say "  $model resident" "Green"
    } catch {
        Say "  WARNING: could not warm $model -- $($_.Exception.Message)" "Yellow"
    }
}

# Report the GPU/CPU split, since a 7B on a 6GB card does not fit whole and
# the spill is the main thing that makes answers feel slow.
try {
    $ps = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/ps" -TimeoutSec 15
    foreach ($m in $ps.models) {
        if ($m.size -gt 0) {
            $pct = [math]::Round(100 * $m.size_vram / $m.size)
            $gb  = [math]::Round($m.size / 1GB, 2)
            if ($pct -ge 99) { Say "  $($m.model): ${gb}GB, 100% GPU" "Green" }
            else { Say "  $($m.model): ${gb}GB, ${pct}% GPU / $(100-$pct)% CPU" "Yellow" }
        }
    }
} catch { Say "  (could not read /api/ps)" "Yellow" }

# ── 3. The API, for the phone ────────────────────────────────────────────────
# 0.0.0.0, not localhost: on a phone "localhost" is the phone itself.
if (Test-Port $ApiPort) {
    Say "Something already listening on $ApiPort; leaving it alone" "Yellow"
} else {
    Say "Starting API on 0.0.0.0:$ApiPort"
    $style = if ($Quiet) { "Hidden" } else { "Minimized" }
    Start-Process -FilePath "python" `
        -ArgumentList "-m","uvicorn","api_server:app","--host","0.0.0.0","--port","$ApiPort" `
        -WorkingDirectory $Root -WindowStyle $style
}

# ── 4. The admin portal ──────────────────────────────────────────────────────
if (-not $NoPortal) {
    if (Test-Port $PortalPort) {
        Say "Something already listening on $PortalPort; leaving it alone" "Yellow"
    } else {
        Say "Starting admin portal on $PortalPort"
        $style = if ($Quiet) { "Hidden" } else { "Minimized" }
        Start-Process -FilePath "python" `
            -ArgumentList "-m","streamlit","run","admin_portal.py",
                          "--server.port","$PortalPort","--server.headless","true",
                          "--browser.gatherUsageStats","false" `
            -WorkingDirectory $Root -WindowStyle $style
    }
}

# ── Report ───────────────────────────────────────────────────────────────────
Start-Sleep -Seconds 6
Say ""
Say "=== ready ===" "Cyan"
if (-not $NoPortal) { Say "  Portal  http://localhost:$PortalPort" }
Say "  Health  http://127.0.0.1:$ApiPort/health"

# Every usable address, not just the first. The Wi-Fi one comes from DHCP and
# changes; reporting a single address is how a stale one ended up baked into
# the app, which then showed Offline against a perfectly healthy server.
$addrs = Get-NetIPAddress -AddressFamily IPv4 |
         Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" }
Say "  API addresses the phone can use:"
foreach ($a in $addrs) {
    $note = ""
    if ($a.InterfaceAlias -like "*Tailscale*") { $note = "  <- stable across networks" }
    elseif ($a.InterfaceAlias -like "*Wi-Fi*") { $note = "  <- DHCP, changes" }
    Say ("    http://{0}:{1}  [{2}]{3}" -f $a.IPAddress, $ApiPort, $a.InterfaceAlias, $note)
}
Say "  Log     $Log"

if (-not (Test-Port $ApiPort)) {
    Say "WARNING: API port $ApiPort is not accepting connections -- check the log" "Red"
}
