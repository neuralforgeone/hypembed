@echo off
setlocal
set SCRATCH=C:\Users\tunay\AppData\Local\Temp\grok-goal-80111e51b701\implementer
set REPO=C:\Users\tunay\Documents\GitHub\hypembed
set LOG1=%SCRATCH%\wasm-build-1.log
set LOG2=%SCRATCH%\wasm-build-2.log
set BINDGEN_LOG=%SCRATCH%\wasm-bindgen.log
set SUMMARY=%SCRATCH%\wasm-verify-summary.log
set TARGET=wasm32-unknown-unknown

mkdir "%SCRATCH%" 2>nul
cd /d "%REPO%"

echo === wasm build run 1 (clean + verbose) === > "%LOG1%"
echo repo=%REPO% >> "%LOG1%"
echo target=%TARGET% >> "%LOG1%"
echo time=%DATE% %TIME% >> "%LOG1%"
echo. >> "%LOG1%"
cargo clean -p hypembed-wasm --target %TARGET% >> "%LOG1%" 2>&1
cargo build -p hypembed-wasm --target %TARGET% -vv >> "%LOG1%" 2>&1
if errorlevel 1 exit /b 1

echo === wasm build run 2 (incremental + verbose) === > "%LOG2%"
echo repo=%REPO% >> "%LOG2%"
echo target=%TARGET% >> "%LOG2%"
echo time=%DATE% %TIME% >> "%LOG2%"
echo. >> "%LOG2%"
cargo build -p hypembed-wasm --target %TARGET% -vv >> "%LOG2%" 2>&1
if errorlevel 1 exit /b 1

findstr /C:"Finished" "%LOG1%" >nul || exit /b 1
findstr /C:"Finished" "%LOG2%" >nul || exit /b 1
findstr /C:"wasm32-unknown-unknown" "%LOG1%" >nul || exit /b 1
findstr /C:"Compiling hypembed-wasm" "%LOG1%" >nul || exit /b 1
findstr /I /C:"error:" "%LOG1%" >nul && exit /b 1
findstr /I /C:"error:" "%LOG2%" >nul && exit /b 1
findstr /I /C:"memmap" "%LOG1%" >nul && exit /b 1
findstr /I /C:"rayon" "%LOG1%" >nul && exit /b 1

echo === wasm-bindgen glue generation === > "%BINDGEN_LOG%"
wasm-bindgen target\%TARGET%\debug\hypembed_wasm.wasm --out-dir examples\bunny-edge --target web --out-name hypembed_wasm >> "%BINDGEN_LOG%" 2>&1
if errorlevel 1 exit /b 1

echo. >> "%BINDGEN_LOG%"
echo === target artifact === >> "%BINDGEN_LOG%"
dir target\%TARGET%\debug\hypembed_wasm.wasm >> "%BINDGEN_LOG%" 2>&1
echo. >> "%BINDGEN_LOG%"
echo === bunny-edge artifacts === >> "%BINDGEN_LOG%"
dir examples\bunny-edge\hypembed_wasm.js >> "%BINDGEN_LOG%" 2>&1
dir examples\bunny-edge\hypembed_wasm_bg.wasm >> "%BINDGEN_LOG%" 2>&1

findstr /C:"__wbg_ptr" examples\bunny-edge\hypembed_wasm.js >nul || exit /b 1
findstr /C:"fnv1a" examples\bunny-edge\hypembed_wasm.js >nul && exit /b 1

echo WASM verify OK > "%SUMMARY%"
echo build1=%LOG1% >> "%SUMMARY%"
echo build2=%LOG2% >> "%SUMMARY%"
echo bindgen=%BINDGEN_LOG% >> "%SUMMARY%"
echo target=%TARGET% >> "%SUMMARY%"
echo artifacts=examples\bunny-edge\hypembed_wasm.js examples\bunny-edge\hypembed_wasm_bg.wasm >> "%SUMMARY%"

echo WASM verify OK. Logs: %LOG1% %LOG2% %BINDGEN_LOG%
exit /b 0