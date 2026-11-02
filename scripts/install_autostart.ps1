<#
    Makes start_srish.ps1 run at every login, so Srish is warm and answering
    before anyone opens the app.

    Install:    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
    Remove:     powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Uninstall

    Uses the per-user Startup folder rather than a scheduled task or a service,
    because it needs no administrator rights and is trivially reversible --
    deleting one .vbs file undoes it completely.

    The .vbs wrapper exists only to launch PowerShell with no visible window;
    a .bat or a shortcut flashes a console at every login.
#>

[CmdletBinding()]
param(
    [switch] $Uninstall,
    [switch] $NoPortal      # autostart the API only, not the portal
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Startup = [Environment]::GetFolderPath("Startup")
$VbsPath = Join-Path $Startup "SrishStartup.vbs"

if ($Uninstall) {
    if (Test-Path $VbsPath) {
        Remove-Item $VbsPath -Force
        Write-Host "Removed $VbsPath" -ForegroundColor Green
        Write-Host "Srish will no longer start at login." -ForegroundColor Gray
    } else {
        Write-Host "Nothing installed at $VbsPath" -ForegroundColor Yellow
    }
    exit 0
}

$script = Join-Path $Root "scripts\start_srish.ps1"
if (-not (Test-Path $script)) {
    Write-Host "FATAL: $script not found" -ForegroundColor Red
    exit 1
}

$extra = if ($NoPortal) { " -NoPortal" } else { "" }

# 0 as the third WindowStyle argument hides the console; False means do not
# block the login sequence waiting for it to finish.
$vbs = @"
' Starts Srish at login with no visible window.
' Installed by scripts\install_autostart.ps1 -- delete this file to disable.
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = "$Root"
shell.Run "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""$script"" -Quiet$extra", 0, False
"@

Set-Content -Path $VbsPath -Value $vbs -Encoding ASCII

Write-Host "Installed." -ForegroundColor Green
Write-Host "  wrapper : $VbsPath"
Write-Host "  runs    : $script$extra"
Write-Host ""
Write-Host "Srish will start at your next login. To start it now without rebooting:" -ForegroundColor Gray
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$script`"" -ForegroundColor Cyan
Write-Host ""
Write-Host "To undo:" -ForegroundColor Gray
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Uninstall" -ForegroundColor Cyan
