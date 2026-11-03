<#
    What is running, and where the model actually sits.

        powershell -ExecutionPolicy Bypass -File scripts\status_srish.ps1
    or
        srish status
#>

[CmdletBinding()]
param(
    [int] $ApiPort    = 8000,
    [int] $PortalPort = 8501
)

$ErrorActionPreference = "Continue"

function Test-Port([int]$Port) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $Port); $c.Close(); return $true
    } catch { return $false }
}

function Mark($ok) { if ($ok) { "[ UP ]" } else { "[DOWN]" } }

Write-Host ""
Write-Host "  SRISH status" -ForegroundColor Cyan
Write-Host "  ------------"

$ollamaUp = Test-Port 11434
$apiUp    = Test-Port $ApiPort
$portalUp = Test-Port $PortalPort

Write-Host ("  {0}  Ollama        127.0.0.1:11434" -f (Mark $ollamaUp)) `
    -ForegroundColor $(if ($ollamaUp) { "Green" } else { "Red" })
Write-Host ("  {0}  API           127.0.0.1:{1}" -f (Mark $apiUp), $ApiPort) `
    -ForegroundColor $(if ($apiUp) { "Green" } else { "Red" })
Write-Host ("  {0}  Admin portal  http://localhost:{1}" -f (Mark $portalUp), $PortalPort) `
    -ForegroundColor $(if ($portalUp) { "Green" } else { "Red" })

# ── Is the bot actually able to answer? ──────────────────────────────────────
# The API stays healthy while Ollama is stopped, so "ready" is the field that
# decides whether the app shows Online or Offline.
if ($apiUp) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 15
        Write-Host ""
        if ($h.ready) {
            Write-Host "  App badge:  ONLINE" -ForegroundColor Green
        } else {
            Write-Host "  App badge:  OFFLINE" -ForegroundColor Red
            foreach ($r in $h.not_ready_because) { Write-Host "    - $r" -ForegroundColor Yellow }
        }
        Write-Host ("    router {0}  present={1}" -f $h.router_model, $h.router_model_present)
        Write-Host ("    chat   {0}  present={1}" -f $h.chat_model, $h.chat_model_present)
        Write-Host ("    knowledge base: {0} diseases" -f $h.kb_entries)
    } catch {
        Write-Host "  /health did not answer: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# ── GPU vs CPU ───────────────────────────────────────────────────────────────
if ($ollamaUp) {
    Write-Host ""
    Write-Host "  Models in memory" -ForegroundColor Cyan
    try {
        $ps = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/ps" -TimeoutSec 15
        if (-not $ps.models) {
            Write-Host "    nothing loaded - the next question pays a load first" -ForegroundColor Yellow
        }
        foreach ($m in $ps.models) {
            $gb = [math]::Round($m.size / 1GB, 2)
            if ($m.size -gt 0) {
                $pct = [math]::Round(100 * $m.size_vram / $m.size)
                $vgb = [math]::Round($m.size_vram / 1GB, 2)
                if ($pct -ge 99) {
                    Write-Host ("    {0}  {1}GB  100% GPU" -f $m.model, $gb) -ForegroundColor Green
                } else {
                    Write-Host ("    {0}  {1}GB  {2}% GPU / {3}% CPU  ({4}GB in VRAM)" -f `
                        $m.model, $gb, $pct, (100 - $pct), $vgb) -ForegroundColor Yellow
                }
            }
            if ($m.expires_at) {
                Write-Host ("      stays loaded until {0}" -f ([datetime]$m.expires_at).ToString("HH:mm:ss"))
            }
        }
    } catch {
        Write-Host "    could not read /api/ps" -ForegroundColor Yellow
    }
}

try {
    $smi = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu `
                        --format=csv,noheader,nounits 2>$null
    if ($smi) {
        $p = $smi -split ","
        Write-Host ""
        Write-Host ("  GPU  {0}" -f $p[0].Trim()) -ForegroundColor Cyan
        Write-Host ("    VRAM {0} / {1} MB   util {2}%   {3}C" -f `
            $p[1].Trim(), $p[2].Trim(), $p[3].Trim(), $p[4].Trim())
    }
} catch { }

# ── Where the phone should point ─────────────────────────────────────────────
# Every usable address, not just the first. The Wi-Fi address comes from DHCP
# and changes, and printing only one is how a stale address ended up baked into
# the app, which then showed Offline against a perfectly healthy server.
Write-Host ""
Write-Host "  Addresses the phone can use" -ForegroundColor Cyan
$addrs = Get-NetIPAddress -AddressFamily IPv4 |
         Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" }
foreach ($a in $addrs) {
    $note = ""
    if ($a.InterfaceAlias -like "*Tailscale*") { $note = "  <- stable across networks" }
    elseif ($a.InterfaceAlias -like "*Wi-Fi*" -or $a.InterfaceAlias -like "*Ethernet*") {
        $note = "  <- DHCP, changes"
    }
    Write-Host ("    http://{0}:{1}   [{2}]{3}" -f $a.IPAddress, $ApiPort, $a.InterfaceAlias, $note)
}
Write-Host ""
Write-Host "  The app probes these automatically. To pin one:" -ForegroundColor Gray
$wifi = ($addrs | Where-Object { $_.InterfaceAlias -notlike "*Tailscale*" } | Select-Object -First 1).IPAddress
Write-Host ("    EXPO_PUBLIC_SRISH_API_URL=http://{0}:{1} npx expo start" -f $wifi, $ApiPort) -ForegroundColor Cyan
Write-Host ""
