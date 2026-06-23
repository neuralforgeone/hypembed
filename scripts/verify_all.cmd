@echo off
setlocal
set SCRATCH=C:\Users\tunay\AppData\Local\Temp\grok-goal-80111e51b701\implementer
set REPO=C:\Users\tunay\Documents\GitHub\hypembed
set TEST_LOG=%SCRATCH%\workspace-test-final.log
set REPO_SCOPE_LOG=%SCRATCH%\repo-scope.log

mkdir "%SCRATCH%" 2>nul
cd /d "%REPO%"

del /q "%SCRATCH%\mcps-absent.log" 2>nul
rmdir /s /q "%REPO%\mcps" 2>nul

echo === repo scope tests === > "%REPO_SCOPE_LOG%"
echo repo=%REPO% >> "%REPO_SCOPE_LOG%"
echo time=%DATE% %TIME% >> "%REPO_SCOPE_LOG%"
echo. >> "%REPO_SCOPE_LOG%"
cargo test repo_scope -- --nocapture >> "%REPO_SCOPE_LOG%" 2>&1
if errorlevel 1 exit /b 1
findstr /C:"test result: ok." "%REPO_SCOPE_LOG%" >nul || exit /b 1

echo === workspace test suite === > "%TEST_LOG%"
echo repo=%REPO% >> "%TEST_LOG%"
echo time=%DATE% %TIME% >> "%TEST_LOG%"
echo. >> "%TEST_LOG%"
cargo test --workspace >> "%TEST_LOG%" 2>&1
if errorlevel 1 exit /b 1

findstr /C:"bunny_edge_handler_wires_wasm_bindgen_glue" "%TEST_LOG%" >nul || exit /b 1
findstr /C:"bunny_edge_wasm_artifact_present" "%TEST_LOG%" >nul || exit /b 1
findstr /C:"test result: ok." "%TEST_LOG%" >nul || exit /b 1
findstr /C:"FAILED" "%TEST_LOG%" >nul && exit /b 1

call "%REPO%\scripts\verify_wasm.cmd"
if errorlevel 1 exit /b 1

call "%REPO%\scripts\verify_cli.cmd" 1
if errorlevel 1 exit /b 1

call "%REPO%\scripts\verify_cli.cmd" 2
if errorlevel 1 exit /b 1

echo ALL VERIFY OK > "%SCRATCH%\verify-all-summary.log"
echo test_log=%TEST_LOG% >> "%SCRATCH%\verify-all-summary.log"
echo repo_scope_log=%REPO_SCOPE_LOG% >> "%SCRATCH%\verify-all-summary.log"

echo ALL VERIFY OK. Evidence in %SCRATCH%
exit /b 0