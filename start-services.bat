@echo off
echo ================================
echo MemoryOS Services Starter
echo ================================

echo Starting MemoryOS API...
start cmd /k "cd MemoryOS\memoryos-pypi && python app.py"

echo Waiting for MemoryOS API to initialize (5 seconds)...
timeout /t 5 /nobreak > nul

echo Starting Bhindi Agent...
start cmd /k "cd MemoryOS\bhindi-agent && npm run dev"

echo.
echo Both services are starting in separate windows.
echo Please wait until both show they are running before testing.
echo.
echo When ready, run: .\test-memory.ps1
echo.
pause
