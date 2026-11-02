@echo off
setlocal
rem One short command for the whole system, so nobody has to remember
rem PowerShell invocations.
rem
rem   srish on        models warm + API for the phone + admin portal
rem   srish off       unload models, stop API and portal  (app shows Offline)
rem   srish status    is it up, and is the model on GPU or CPU
rem   srish portal    just the admin portal
rem   srish api       just the API, no portal
rem   srish restart   off then on

cd /d "%~dp0"
set PS=powershell -NoProfile -ExecutionPolicy Bypass -File

if "%~1"=="" goto :on
if /i "%~1"=="on"      goto :on
if /i "%~1"=="start"   goto :on
if /i "%~1"=="off"     goto :off
if /i "%~1"=="stop"    goto :off
if /i "%~1"=="status"  goto :status
if /i "%~1"=="portal"  goto :portal
if /i "%~1"=="api"     goto :api
if /i "%~1"=="restart" goto :restart
if /i "%~1"=="help"    goto :help
if /i "%~1"=="-h"      goto :help
if /i "%~1"=="--help"  goto :help

echo Unknown command: %~1
echo.
goto :help

:on
%PS% "scripts\start_srish.ps1" %2 %3 %4 %5
goto :eof

:off
%PS% "scripts\stop_srish.ps1" %2 %3 %4 %5
goto :eof

:restart
%PS% "scripts\stop_srish.ps1" -Quiet
%PS% "scripts\start_srish.ps1" %2 %3 %4 %5
goto :eof

:portal
rem Runs in the foreground, same as typing: streamlit run admin_portal.py
python -m streamlit run admin_portal.py
goto :eof

:api
%PS% "scripts\start_srish.ps1" -NoPortal %2 %3 %4 %5
goto :eof

:status
%PS% "scripts\status_srish.ps1"
goto :eof

:help
echo.
echo   srish on        start everything ^(models warm, API, portal^)
echo   srish off       unload models and stop servers - app shows Offline
echo   srish restart   off then on
echo   srish status    what is running, and GPU vs CPU split
echo   srish portal    admin portal only
echo   srish api       API only, no portal
echo.
echo   Portal   http://localhost:8501
echo   API      http://localhost:8000      ^(phone uses your LAN IP^)
echo.
goto :eof
