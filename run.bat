@echo off
rem One command to set up and start laya-snake on Windows.
setlocal
set DIR=%~dp0
where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%DIR%play.py" %*
) else (
    py "%DIR%play.py" %*
)
