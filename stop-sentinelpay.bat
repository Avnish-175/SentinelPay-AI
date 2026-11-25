@echo off
setlocal EnableDelayedExpansion

title SentinelPay AI Shutdown
echo ===============================================================================
echo                     SENTINELPAY AI - SERVICE SHUTDOWN                          
echo ===============================================================================
echo.

:: -----------------------------------------------------------------------------
:: STOP FASTAPI (PORT 8000)
:: -----------------------------------------------------------------------------
echo [*] Checking for process listening on Port 8000 (FastAPI ML Engine)...
for /f "tokens=5" %%a in ('netstat -a -n -o ^| findstr /r ":8000.*LISTENING"') do (
    set "PID_8000=%%a"
    echo [*] Stopping FastAPI ML Engine process PID !PID_8000!...
    taskkill /PID !PID_8000! /T /F >nul 2>&1
    if not errorlevel 1 (
        echo [OK] FastAPI ML Engine stopped.
    )
)
if "!PID_8000!"=="" (
    echo [INFO] No process listening on Port 8000.
)

:: -----------------------------------------------------------------------------
:: STOP SPRING BOOT (PORT 8081)
:: -----------------------------------------------------------------------------
echo.
echo [*] Checking for process listening on Port 8081 (Spring Boot Backend)...
for /f "tokens=5" %%a in ('netstat -a -n -o ^| findstr /r ":8081.*LISTENING"') do (
    set "PID_8081=%%a"
    echo [*] Stopping Spring Boot Backend process PID !PID_8081!...
    taskkill /PID !PID_8081! /T /F >nul 2>&1
    if not errorlevel 1 (
        echo [OK] Spring Boot Backend stopped.
    )
)
if "!PID_8081!"=="" (
    echo [INFO] No process listening on Port 8081.
)

:: -----------------------------------------------------------------------------
:: STOP REACT FRONTEND (PORT 5173)
:: -----------------------------------------------------------------------------
echo.
echo [*] Checking for process listening on Port 5173 (React Frontend)...
for /f "tokens=5" %%a in ('netstat -a -n -o ^| findstr /r ":5173.*LISTENING"') do (
    set "PID_5173=%%a"
    echo [*] Stopping React Frontend process PID !PID_5173!...
    taskkill /PID !PID_5173! /T /F >nul 2>&1
    if not errorlevel 1 (
        echo [OK] React Frontend stopped.
    )
)
if "!PID_5173!"=="" (
    echo [INFO] No process listening on Port 5173.
)

:: -----------------------------------------------------------------------------
:: CLEAN UP TITLED CMD CONSOLES LAUNCHED BY SCRIPT
:: -----------------------------------------------------------------------------
echo.
echo [*] Closing SentinelPay console windows...
taskkill /FI "WINDOWTITLE eq SentinelPay - FastAPI ML Engine [Port 8000]*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq SentinelPay - Spring Boot Backend [Port 8081]*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq SentinelPay - React Frontend [Port 5173]*" /T /F >nul 2>&1

echo.
echo ===============================================================================
echo SentinelPay AI services have been stopped.
echo ===============================================================================
echo.
pause
