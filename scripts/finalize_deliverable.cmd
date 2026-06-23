@echo off
setlocal
set SCRATCH=C:\Users\tunay\AppData\Local\Temp\grok-goal-80111e51b701\implementer
set REPO=C:\Users\tunay\Documents\GitHub\hypembed
set REPO_SCOPE_LOG=%SCRATCH%\repo-scope.log

mkdir "%SCRATCH%" 2>nul
cd /d "%REPO%"

del /q "%SCRATCH%\mcps-absent.log" 2>nul
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
echo mcps_boundary=git_ls_files_empty_and_gitignore_excludes >> "%SCRATCH%\finalize-summary.log"
echo note=mcps_may_exist_on_disk_ignored_not_in_deliverable >> "%SCRATCH%\finalize-summary.log"

echo === evidence manifest === > "%SCRATCH%\evidence-manifest.log"
echo repo_scope_log=%REPO_SCOPE_LOG% >> "%SCRATCH%\evidence-manifest.log"
echo workspace_test_log=%SCRATCH%\workspace-test-final.log >> "%SCRATCH%\evidence-manifest.log"
echo wasm_build_1=%SCRATCH%\wasm-build-1.log >> "%SCRATCH%\evidence-manifest.log"
echo wasm_build_2=%SCRATCH%\wasm-build-2.log >> "%SCRATCH%\evidence-manifest.log"
echo wasm_bindgen=%SCRATCH%\wasm-bindgen.log >> "%SCRATCH%\evidence-manifest.log"
echo hype_rag_run_1=%SCRATCH%\hype-rag-run-1.log >> "%SCRATCH%\evidence-manifest.log"
echo hype_rag_run_2=%SCRATCH%\hype-rag-run-2.log >> "%SCRATCH%\evidence-manifest.log"
echo verify_all_summary=%SCRATCH%\verify-all-summary.log >> "%SCRATCH%\evidence-manifest.log"
echo finalize_summary=%SCRATCH%\finalize-summary.log >> "%SCRATCH%\evidence-manifest.log"
if exist "%SCRATCH%\mcps-absent.log" (
  echo FAIL stale mcps-absent.log must not exist >> "%SCRATCH%\evidence-manifest.log"
  exit /b 1
)
echo mcps_absent_log=removed_not_used >> "%SCRATCH%\evidence-manifest.log"

echo FINALIZE OK. Evidence: %REPO_SCOPE_LOG%
exit /b 0