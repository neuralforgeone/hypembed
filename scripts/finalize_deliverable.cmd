@echo off
setlocal
set SCRATCH=C:\Users\tunay\AppData\Local\Temp\grok-goal-80111e51b701\implementer
set REPO=C:\Users\tunay\Documents\GitHub\hypembed
set REPO_SCOPE_LOG=%SCRATCH%\repo-scope.log

mkdir "%SCRATCH%" 2>nul
cd /d "%REPO%"

rmdir /s /q "%REPO%\mcps" 2>nul

call "%REPO%\scripts\verify_all.cmd"
if errorlevel 1 exit /b 1

rmdir /s /q "%REPO%\mcps" 2>nul

echo. >> "%REPO_SCOPE_LOG%"
echo === git ls-files mcps/ === >> "%REPO_SCOPE_LOG%"
git ls-files -- mcps/ >> "%REPO_SCOPE_LOG%" 2>&1
echo. >> "%REPO_SCOPE_LOG%"
echo === git status --ignored -- mcps/ === >> "%REPO_SCOPE_LOG%"
git status --ignored -- mcps/ >> "%REPO_SCOPE_LOG%" 2>&1

echo FINALIZE OK > "%SCRATCH%\finalize-summary.log"
echo repo_scope_log=%REPO_SCOPE_LOG% >> "%SCRATCH%\finalize-summary.log"
echo verify_summary=%SCRATCH%\verify-all-summary.log >> "%SCRATCH%\finalize-summary.log"

echo FINALIZE OK. Evidence: %REPO_SCOPE_LOG%
exit /b 0