@echo off
setlocal
cd /d "%~dp0\.."

cargo build -p hypembed-wasm --target wasm32-unknown-unknown
if errorlevel 1 exit /b 1

wasm-bindgen target\wasm32-unknown-unknown\debug\hypembed_wasm.wasm --out-dir examples\bunny-edge --target web --out-name hypembed_wasm
if errorlevel 1 exit /b 1

echo Generated examples\bunny-edge\hypembed_wasm.js and hypembed_wasm_bg.wasm
exit /b 0