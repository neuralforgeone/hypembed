@echo off
setlocal
set SCRATCH=C:\Users\tunay\AppData\Local\Temp\grok-goal-80111e51b701\implementer
set RUN=%1
if "%RUN%"=="" set RUN=1
set LOG=%SCRATCH%\hype-rag-run-%RUN%.log
set REPO=C:\Users\tunay\Documents\GitHub\hypembed
set VERIFY_ROOT=%SCRATCH%\hype-rag-verify-%RUN%

mkdir "%SCRATCH%" 2>nul
cd /d "%REPO%"

echo === hype-rag CLI verify run %RUN% === > "%LOG%"
cargo build -p hype-rag >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

if exist "%VERIFY_ROOT%" rmdir /s /q "%VERIFY_ROOT%"
mkdir "%VERIFY_ROOT%\model"
mkdir "%VERIFY_ROOT%\docs"

cargo run --example write_tiny_model -- %VERIFY_ROOT%\model >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

cargo run -p hype-rag -- --data-dir "%VERIFY_ROOT%\.hype-rag" init --model-dir "%VERIFY_ROOT%\model" >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

echo Rust embeddings run locally without Python dependencies.> "%VERIFY_ROOT%\docs\doc1.md"
echo Machine learning semantic search uses vector similarity.> "%VERIFY_ROOT%\docs\doc2.txt"
echo Bake bread at 180 degrees for forty minutes.> "%VERIFY_ROOT%\docs\doc3.md"

cargo run -p hype-rag -- --data-dir "%VERIFY_ROOT%\.hype-rag" index "%VERIFY_ROOT%\docs" --recursive >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

cargo run -p hype-rag -- --data-dir "%VERIFY_ROOT%\.hype-rag" search "rust embedding" >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

findstr /C:"score=" "%LOG%" >nul || exit /b 1
findstr /C:"path=" "%LOG%" >nul || exit /b 1

echo CLI verify run %RUN% OK. Log: %LOG%
exit /b 0