<#
    Makes this laptop answer on an address an already-built app is calling.

    A built APK cannot discover the laptop the way Expo does in development --
    whatever address was compiled in at build time is the only one it will try.
    Builds made before 2026-11-03 11:17 target 192.168.1.14, and this laptop's
    DHCP lease has since moved to 192.168.1.39, so those builds reach nothing.

    Rather than rebuild the app, give the laptop that address as well. A network
    adapter can hold several, so the API becomes reachable on both at once.

    NEEDS AN ADMINISTRATOR PowerShell.

        powershell -ExecutionPolicy Bypass -File scripts\add_app_ip.ps1
        powershell -ExecutionPolicy Bypass -File scripts\add_app_ip.ps1 -Remove
#>

[CmdletBinding()]
param(
    [string] $IPAddress = "192.168.1.14",
    [string] $InterfaceAlias = "Wi-Fi",
    [int]    $PrefixLength = 24,
    [int]    $ApiPort = 8000,
    [switch] $Remove
)

$ErrorActionPreference = "Stop"

function Fail($m) { Write-Host "  $m" -ForegroundColor Red; exit 1 }
function Ok($m)   { Write-Host "  $m" -ForegroundColor Green }
function Note($m) { Write-Host "  $m" -ForegroundColor Gray }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host ""
    Fail @"
This needs an Administrator PowerShell.

  1. Press Start, type: powershell
  2. Right-click 'Windows PowerShell' -> Run as administrator
  3. Run:
     cd "$(Split-Path -Parent $PSScriptRoot)"
     powershell -ExecutionPolicy Bypass -File scripts\add_app_ip.ps1
"@
}

Write-Host ""

# ── Remove ───────────────────────────────────────────────────────────────────
if ($Remove) {
    $existing = Get-NetIPAddress -IPAddress $IPAddress -ErrorAction SilentlyContinue
    if (-not $existing) { Note "$IPAddress is not configured on this machine."; exit 0 }
    Remove-NetIPAddress -IPAddress $IPAddress -Confirm:$false
    Ok "Removed $IPAddress."
    exit 0
}

# ── Already there? ───────────────────────────────────────────────────────────
$existing = Get-NetIPAddress -IPAddress $IPAddress -ErrorAction SilentlyContinue
if ($existing) {
    Ok "$IPAddress is already on this machine ($($existing.InterfaceAlias))."
} else {
    # Someone else on the network may hold it. Adding it anyway would give two
    # machines the same address and break both.
    Note "Checking nobody else is using $IPAddress ..."
    if (Test-Connection -ComputerName $IPAddress -Count 2 -Quiet -ErrorAction SilentlyContinue) {
        Fail @"
$IPAddress already answers on this network -- another device has it.

Claiming it would break that device and this one. Instead, rebuild the app
with the right address, or set it before building:
  app.json  ->  expo.extra.srishApiUrl = "http://<this laptop>:$ApiPort"
"@
    }
    Ok "Free."

    $nic = Get-NetAdapter -Name $InterfaceAlias -ErrorAction SilentlyContinue
    if (-not $nic) {
        Note "Adapters on this machine:"
        Get-NetAdapter | ForEach-Object { Note "  - $($_.Name)  [$($_.Status)]" }
        Fail "No adapter named '$InterfaceAlias'. Re-run with -InterfaceAlias '<name>'."
    }

    # SkipAsSource keeps outbound traffic on the DHCP address: this extra one
    # exists only so the app's requests arrive, not to originate connections.
    New-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $IPAddress `
        -PrefixLength $PrefixLength -SkipAsSource $true | Out-Null
    Ok "Added $IPAddress to $InterfaceAlias."
}

# ── Does the API actually answer there? ──────────────────────────────────────
Write-Host ""
Note "Checking the API responds on $IPAddress ..."
Start-Sleep -Seconds 2
try {
    $h = Invoke-RestMethod -Uri "http://${IPAddress}:$ApiPort/health" -TimeoutSec 15
    if ($h.ready) { Ok "http://${IPAddress}:$ApiPort  ->  ready. The built app will work." }
    else {
        Note "Reachable, but not ready: $($h.not_ready_because -join '; ')"
        Note "Run 'srish on' to start the models."
    }
} catch {
    Note "No answer yet on http://${IPAddress}:$ApiPort"
    Note "Start the API with 'srish on', then re-run this script to confirm."
}

Write-Host ""
Note "This address persists across reboots."
Note "To undo:  powershell -ExecutionPolicy Bypass -File scripts\add_app_ip.ps1 -Remove"
Write-Host ""
