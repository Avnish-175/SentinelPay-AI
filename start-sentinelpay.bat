@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ============================================================================
:: SentinelPay AI - Unified Windows Startup Script
:: ============================================================================

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "ML_DIR=%ROOT_DIR%\ml-engine"
set "BACKEND_DIR=%ROOT_DIR%\backend"
set "FRONTEND_DIR=%ROOT_DIR%\frontend"

set "ML_PORT=8000"
set "BACKEND_PORT=8081"
set "FRONTEND_PORT=5173"
set "PG_PORT=5432"

echo ============================================================================
echo   SentinelPay AI - Starting Services
echo   Project Root: %ROOT_DIR%
echo ============================================================================
echo.

:: 1. Verify project directories
if not exist "%ML_DIR%\main.py" (
    echo [ERROR] ML Engine entry point not found at %ML_DIR%\main.py
    goto :fail
)
if not exist "%BACKEND_DIR%\pom.xml" (
    echo [ERROR] Backend pom.xml not found at %BACKEND_DIR%\pom.xml
    goto :fail
)
if not exist "%FRONTEND_DIR%\package.json" (
    echo [ERROR] Frontend package.json not found at %FRONTEND_DIR%\package.json
    goto :fail
)

:: 2. Detect required binaries
echo [CHECK] Detecting runtime dependencies...

call :check_tool java "Java JDK"
if errorlevel 1 goto :fail

call :check_tool python "Python"
if errorlevel 1 goto :fail

call :check_tool node "Node.js"
if errorlevel 1 goto :fail

call :check_tool npm "npm"
if errorlevel 1 goto :fail

:: Detect Maven across PATH, MAVEN_HOME, M2_HOME, or .m2 wrapper distributions
set "MVN_CMD="
where mvn >nul 2>&1
if not errorlevel 1 set "MVN_CMD=mvn"

if not defined MVN_CMD (
    where mvn.cmd >nul 2>&1
    if not errorlevel 1 set "MVN_CMD=mvn.cmd"
)

if not defined MVN_CMD (
    if defined MAVEN_HOME (
        if exist "%MAVEN_HOME%\bin\mvn.cmd" set "MVN_CMD=%MAVEN_HOME%\bin\mvn.cmd"
    )
)

if not defined MVN_CMD (
    if defined M2_HOME (
        if exist "%M2_HOME%\bin\mvn.cmd" set "MVN_CMD=%M2_HOME%\bin\mvn.cmd"
    )
)

if not defined MVN_CMD (
    if exist "%USERPROFILE%\.m2\wrapper\dists" (
        for /f "delims=" %%F in ('dir /s /b "%USERPROFILE%\.m2\wrapper\dists\mvn.cmd" 2^>nul') do (
            if not defined MVN_CMD if exist "%%F" set "MVN_CMD=%%F"
        )
    )
)

if not defined MVN_CMD (
    echo.
    echo [ERROR] Apache Maven [mvn] was not found on PATH or in standard Maven directories.
    echo         Please install Apache Maven and add its bin folder to system PATH,
    echo         or set the MAVEN_HOME environment variable.
    goto :fail
)

echo [OK] All runtime dependencies verified.
echo [INFO] Using Maven: %MVN_CMD%
echo.

:: 3. Check PostgreSQL on localhost:5432
echo [CHECK] Verifying PostgreSQL availability on localhost:%PG_PORT%...
call :check_port %PG_PORT%
if errorlevel 1 (
    echo.
    echo [ERROR] PostgreSQL is not reachable on localhost:%PG_PORT%.
    echo         The Spring Boot backend requires a running PostgreSQL instance.
    echo         Database URL: jdbc:postgresql://localhost:5432/sentinelpay
    echo         Please start your PostgreSQL service and try again.
    goto :fail
)
echo [OK] PostgreSQL is reachable on port %PG_PORT%.
echo.

:: 4. Start Python ML Engine (FastAPI on port 8000)
call :check_port %ML_PORT%
if not errorlevel 1 (
    echo [SKIP] ML Engine is already running on port %ML_PORT%.
) else (
    set "VENV_ACTIVATE="
    if exist "%ML_DIR%\.venv\Scripts\activate.bat" set "VENV_ACTIVATE=%ML_DIR%\.venv\Scripts\activate.bat"
    if not defined VENV_ACTIVATE if exist "%ML_DIR%\venv\Scripts\activate.bat" set "VENV_ACTIVATE=%ML_DIR%\venv\Scripts\activate.bat"
    if not defined VENV_ACTIVATE if exist "%ROOT_DIR%\.venv\Scripts\activate.bat" set "VENV_ACTIVATE=%ROOT_DIR%\.venv\Scripts\activate.bat"
    if not defined VENV_ACTIVATE if exist "%ROOT_DIR%\venv\Scripts\activate.bat" set "VENV_ACTIVATE=%ROOT_DIR%\venv\Scripts\activate.bat"

    echo [START] Launching Python ML Engine on port %ML_PORT%...
    if defined VENV_ACTIVATE (
        echo [INFO] Activating virtual environment: %VENV_ACTIVATE%
        start "SentinelPay ML Engine" cmd /k "cd /d "%ML_DIR%" && call "%VENV_ACTIVATE%" && python main.py"
    ) else (
        echo [INFO] No virtual environment folder found; using system Python.
        start "SentinelPay ML Engine" cmd /k "cd /d "%ML_DIR%" && python main.py"
    )
)

echo [WAIT] Waiting for ML Engine health endpoint [http://localhost:%ML_PORT%/health]...
call :wait_http "http://localhost:%ML_PORT%/health" 45
if errorlevel 1 (
    echo.
    echo [ERROR] ML Engine failed to respond on http://localhost:%ML_PORT%/health within 45s.
    echo         Please inspect the "SentinelPay ML Engine" console window for error messages.
    goto :fail
)
echo [OK] ML Engine is healthy.
echo.

:: 5. Start Spring Boot Backend (JAR or Maven on port 8081)
call :check_port %BACKEND_PORT%
if not errorlevel 1 (
    echo [SKIP] Spring Boot Backend is already running on port %BACKEND_PORT%.
) else (
    set "BACKEND_JAR="
    if exist "%BACKEND_DIR%\target\sentinelpay-backend-1.0.0.jar" set "BACKEND_JAR=%BACKEND_DIR%\target\sentinelpay-backend-1.0.0.jar"
    if not defined BACKEND_JAR (
        for %%J in ("%BACKEND_DIR%\target\*.jar") do (
            if not defined BACKEND_JAR set "BACKEND_JAR=%%~fJ"
        )
    )

    if defined BACKEND_JAR (
        echo [START] Launching Spring Boot Backend via packaged JAR: !BACKEND_JAR!...
        start "SentinelPay Backend" cmd /k "cd /d "%BACKEND_DIR%" && java -jar "!BACKEND_JAR!""
    ) else (
        echo [START] Launching Spring Boot Backend via Maven on port %BACKEND_PORT%...
        start "SentinelPay Backend" cmd /k "cd /d "%BACKEND_DIR%" && "%MVN_CMD%" spring-boot:run"
    )
)

echo [WAIT] Waiting for Spring Boot Backend on port %BACKEND_PORT%...
call :wait_port %BACKEND_PORT% 90
if errorlevel 1 (
    echo.
    echo [ERROR] Spring Boot Backend did not start listening on port %BACKEND_PORT% within 90s.
    echo         Please inspect the "SentinelPay Backend" console window for build or startup errors.
    goto :fail
)
echo [OK] Spring Boot Backend is ready.
echo.

:: 6. Start React Frontend (Vite on port 5173)
call :check_port %FRONTEND_PORT%
if not errorlevel 1 (
    echo [SKIP] React Frontend is already running on port %FRONTEND_PORT%.
) else (
    echo [START] Launching React Frontend on port %FRONTEND_PORT%...
    start "SentinelPay Frontend" cmd /k "cd /d "%FRONTEND_DIR%" && npm run dev"
)

echo [WAIT] Waiting for React Frontend [http://localhost:%FRONTEND_PORT%]...
call :wait_http "http://localhost:%FRONTEND_PORT%" 30
if errorlevel 1 (
    echo.
    echo [ERROR] React Frontend did not respond on http://localhost:%FRONTEND_PORT% within 30s.
    echo         Please inspect the "SentinelPay Frontend" console window for details.
    goto :fail
)
echo [OK] React Frontend is ready.
echo.

:: 7. Launch browser
echo ============================================================================
echo   SentinelPay AI is running successfully!
echo   - Frontend:  http://localhost:%FRONTEND_PORT%
echo   - Backend:   http://localhost:%BACKEND_PORT%
echo   - ML Engine: http://localhost:%ML_PORT%/health
echo ============================================================================
start "" "http://localhost:%FRONTEND_PORT%"
goto :eof

:: ============================================================================
:: Subroutines
:: ============================================================================

:check_tool
where %~1 >nul 2>&1
if errorlevel 1 (
    echo [ERROR] %~2 [%~1] was not found on system PATH.
    echo         Please install %~2 and ensure it is included in your system PATH.
    exit /b 1
)
exit /b 0

:check_port
powershell -NoProfile -Command "$client = New-Object System.Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', %~1); $client.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
exit /b %errorlevel%

:wait_port
set /a "_p_elapsed=0"
:_wait_port_loop
call :check_port %~1
if not errorlevel 1 exit /b 0
if %_p_elapsed% GEQ %~2 exit /b 1
timeout /t 2 /nobreak >nul
set /a "_p_elapsed+=2"
goto :_wait_port_loop

:wait_http
set /a "_h_elapsed=0"
:_wait_http_loop
where curl.exe >nul 2>&1
if not errorlevel 1 (
    curl.exe -s -f -o nul --connect-timeout 2 "%~1" >nul 2>&1
    if not errorlevel 1 exit /b 0
) else (
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%~1' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 exit /b 0
)
if %_h_elapsed% GEQ %~2 exit /b 1
timeout /t 2 /nobreak >nul
set /a "_h_elapsed+=2"
goto :_wait_http_loop

:fail
echo.
echo ============================================================================
echo   Startup aborted due to an error.
echo ============================================================================
pause
exit /b 1
