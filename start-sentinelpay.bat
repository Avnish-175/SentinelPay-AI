@echo off
setlocal EnableDelayedExpansion

title SentinelPay AI Launcher
echo ===============================================================================
echo                     SENTINELPAY AI - FULL STACK LAUNCHER
echo ===============================================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Refresh PATH to discover node, npm, python, and postgresql
set "PATH=%PATH%;%ProgramFiles%\nodejs;%LOCALAPPDATA%\Programs\Python\Python314\Scripts;%LOCALAPPDATA%\Programs\Python\Python314;%ProgramFiles%\PostgreSQL\19\bin"

:: -----------------------------------------------------------------------------
:: 1. CHECK POSTGRESQL SERVICE
:: -----------------------------------------------------------------------------
echo [*] Checking PostgreSQL service...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$svc = Get-Service *postgres* -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq 'Running' }; if ($svc) { exit 0 } else { exit 1 }"

if errorlevel 1 (
    echo [ERROR] PostgreSQL service is not currently running!
    echo         Please start the PostgreSQL Windows service before running SentinelPay AI.
    echo.
    pause
    exit /b 1
)
echo [OK] PostgreSQL service is running.

:: -----------------------------------------------------------------------------
:: 2. DETECT MAVEN EXECUTABLE
:: -----------------------------------------------------------------------------
set "MVN_CMD="
where mvn >nul 2>&1
if not errorlevel 1 (
    set "MVN_CMD=mvn"
) else if exist "%ProgramFiles%\JetBrains\IntelliJ IDEA 2026.2.2\plugins\maven-plugin\lib\maven3\bin\mvn.cmd" (
    set "MVN_CMD=%ProgramFiles%\JetBrains\IntelliJ IDEA 2026.2.2\plugins\maven-plugin\lib\maven3\bin\mvn.cmd"
) else if exist "%USERPROFILE%\.m2\wrapper\dists\apache-maven-3.9.16-bin\5grr65jo27hi51sujmtcldfovl\apache-maven-3.9.16\bin\mvn.cmd" (
    set "MVN_CMD=%USERPROFILE%\.m2\wrapper\dists\apache-maven-3.9.16-bin\5grr65jo27hi51sujmtcldfovl\apache-maven-3.9.16\bin\mvn.cmd"
)

if "%MVN_CMD%"=="" (
    echo [ERROR] Maven executable was not found on your system!
    echo         Please install Apache Maven and add it to your system PATH.
    echo.
    pause
    exit /b 1
)
echo [OK] Maven found: "%MVN_CMD%"

:: -----------------------------------------------------------------------------
:: 3. DETECT NODE.JS & NPM
:: -----------------------------------------------------------------------------
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not found in PATH!
    echo         Please install Node.js LTS from https://nodejs.org/.
    echo.
    pause
    exit /b 1
)
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm is not found in PATH!
    echo         Please ensure npm is installed and added to your PATH.
    echo.
    pause
    exit /b 1
)
echo [OK] Node.js and npm verified.

:: -----------------------------------------------------------------------------
:: 4. START FASTAPI ML ENGINE (PORT 8000)
:: -----------------------------------------------------------------------------
echo.
echo [*] Checking FastAPI ML Engine (Port 8000)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$c = Test-NetConnection -ComputerName localhost -Port 8000 -WarningAction SilentlyContinue; if ($c.TcpTestSucceeded) { exit 0 } else { exit 1 }"

if not errorlevel 1 (
    echo [INFO] Port 8000 is already active. Preserving existing FastAPI ML Engine process.
) else (
    echo [*] Starting FastAPI ML Engine in a new window...
    if exist "%SCRIPT_DIR%ml-engine\.venv\Scripts\activate.bat" (
        start "SentinelPay - FastAPI ML Engine [Port 8000]" /D "%SCRIPT_DIR%ml-engine" cmd /k "call "%SCRIPT_DIR%ml-engine\.venv\Scripts\activate.bat" && uvicorn main:app --host 0.0.0.0 --port 8000"
    ) else if exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" (
        start "SentinelPay - FastAPI ML Engine [Port 8000]" /D "%SCRIPT_DIR%ml-engine" cmd /k "call "%SCRIPT_DIR%.venv\Scripts\activate.bat" && uvicorn main:app --host 0.0.0.0 --port 8000"
    ) else (
        start "SentinelPay - FastAPI ML Engine [Port 8000]" /D "%SCRIPT_DIR%ml-engine" cmd /k "uvicorn main:app --host 0.0.0.0 --port 8000"
    )
)

:: -----------------------------------------------------------------------------
:: 5. START SPRING BOOT BACKEND (PORT 8081)
:: -----------------------------------------------------------------------------
echo.
echo [*] Checking Spring Boot Backend (Port 8081)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$c = Test-NetConnection -ComputerName localhost -Port 8081 -WarningAction SilentlyContinue; if ($c.TcpTestSucceeded) { exit 0 } else { exit 1 }"

if not errorlevel 1 (
    echo [INFO] Port 8081 is already active. Preserving existing Spring Boot Backend process.
) else (
    echo [*] Starting Spring Boot Backend in a new window...
    start "SentinelPay - Spring Boot Backend [Port 8081]" /D "%SCRIPT_DIR%backend" cmd /k ""%MVN_CMD%" spring-boot:run -Dspring-boot.run.arguments=--server.port=8081"
)

:: -----------------------------------------------------------------------------
:: 6. START REACT FRONTEND (PORT 5173)
:: -----------------------------------------------------------------------------
echo.
echo [*] Checking React Frontend (Port 5173)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$c = Test-NetConnection -ComputerName localhost -Port 5173 -WarningAction SilentlyContinue; if ($c.TcpTestSucceeded) { exit 0 } else { exit 1 }"

if not errorlevel 1 (
    echo [INFO] Port 5173 is already active. Preserving existing React Frontend process.
) else (
    echo [*] Starting React Frontend in a new window...
    start "SentinelPay - React Frontend [Port 5173]" /D "%SCRIPT_DIR%frontend" cmd /k "npm run dev"
)

:: -----------------------------------------------------------------------------
:: 7. VERIFY ALL SERVICES & LAUNCH BROWSER
:: -----------------------------------------------------------------------------
echo.
echo [*] Waiting for FastAPI (8000), Spring Boot (8081), and React (5173) to become ready...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ports = @(8000, 8081, 5173); $maxAttempts = 45; $attempt = 0; " ^
    "while ($attempt -lt $maxAttempts) { " ^
    "  $attempt++; $allUp = $true; " ^
    "  foreach ($p in $ports) { " ^
    "    try { $tc = New-Object System.Net.Sockets.TcpClient; $ok = $tc.ConnectAsync('127.0.0.1', $p).Wait(300); $tc.Close(); if (-not $ok) { $allUp = $false; break } } catch { $allUp = $false; break } " ^
    "  } " ^
    "  if ($allUp) { exit 0 } " ^
    "  Start-Sleep -Seconds 1 " ^
    "} " ^
    "exit 1"

if errorlevel 1 (
    echo [WARN] One or more services did not respond within 45 seconds.
    echo        Check the individual terminal windows for any startup errors.
    echo.
    pause
    exit /b 1
)

echo [OK] All services are listening! Opening http://localhost:5173 in default browser...
start "" "http://localhost:5173"

echo.
echo ===============================================================================
echo SentinelPay AI services launched successfully!
echo   - FastAPI ML Engine:    http://localhost:8000/health
echo   - Spring Boot Backend:  http://localhost:8081/api/v1/transactions
echo   - React UI Dashboard:   http://localhost:5173
echo.
echo Note: Logs remain active in their respective persistent terminal windows.
echo To shut down all services gracefully, double-click 'stop-sentinelpay.bat'.
echo ===============================================================================
echo.
pause
