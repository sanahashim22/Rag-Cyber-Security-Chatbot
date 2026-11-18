<#
.SYNOPSIS
  Launches the dashboard pointed at a REMOTE Ollama server (e.g. one running
  on a GPU machine, exposed via `ngrok http 11434` on that machine) instead
  of your local CPU Ollama - so you can compare ranking times GPU vs CPU
  using the exact same app/code.

.USAGE
  On the GPU machine:
    ollama serve
    ollama pull phi4-mini          # or whichever model you're testing
    ngrok http 11434               # copy the https://....ngrok-free.dev URL it prints

  Here, on this machine:
    .\run_gpu_test.ps1 -OllamaHost "https://xxxx.ngrok-free.dev"

  For plain local testing (CPU, as usual), don't use this script - just run:
    streamlit run dashboard_app.py
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$OllamaHost
)

# RAG_OLLAMA_HOST, not OLLAMA_HOST - the latter is Ollama's own server-side
# bind-address variable; reusing it here would collide with it on machines
# where OLLAMA_HOST is already set system-wide for the server itself.
$env:RAG_OLLAMA_HOST = $OllamaHost
Write-Host "Using remote Ollama host: $env:RAG_OLLAMA_HOST" -ForegroundColor Cyan
python -m streamlit run dashboard_app.py
