# run_tests.ps1 - Sets up correct DLL path and runs pytest for DeepFake Sentinel 2030
# Must be run from the project root: deep_learning/

$ErrorActionPreference = "Stop"

$torchLib = "C:\Users\dhanu\AppData\Local\Python\pythoncore-3.11-64\Lib\site-packages\torch\lib"
$pytest   = "C:\Users\dhanu\odysseus\venv\Scripts\python.exe"

# Add torch DLLs to PATH so torch._C can load
$env:PATH = "$torchLib;$env:PATH"
# Add project root to PYTHONPATH
$env:PYTHONPATH = $PSScriptRoot

Write-Host "Running DeepFake Sentinel 2030 tests..." -ForegroundColor Cyan
& $pytest -m pytest tests/test_backend.py -v
