# Local LLM

A privacy-first Python LLM service that runs entirely on your machine. It uses a local GGUF model through `llama-cpp-python`; it does not call OpenAI, cloud APIs, telemetry services, or download anything at runtime.

## Setup

1. Create an environment and install the local inference dependency:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Put the Qwen3 model at `models/Qwen3-8B-Q4_K_M.gguf`. Download this file separately before going offline, or point to another local file with `LOCAL_LLM_MODEL`.

3. Start the server:

   ```powershell
   python local_llm.py
   ```

The server binds to `127.0.0.1` only. Check it with `http://127.0.0.1:8000/health`.

## Use

One-shot CLI:

```powershell
python local_llm.py --prompt "Explain recursion in one paragraph."
```

HTTP request:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/generate `
  -ContentType 'application/json' `
  -Body '{"prompt":"Write a haiku about local software","max_tokens":80}'
```

Useful environment variables are `LOCAL_LLM_MODEL`, `LOCAL_LLM_HOST`, `LOCAL_LLM_PORT`, `LOCAL_LLM_CONTEXT`, and `LOCAL_LLM_THREADS`.

## Verify

```powershell
python -m unittest discover -s tests -v
```

## GitHub connection

This folder currently has no Git remote. After creating or choosing the GitHub repository, connect it with:

```powershell
git init
git add .
git commit -m "Build offline local LLM service"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the placeholder URL with your repository URL. Do not commit model weights; they are ignored by Git.
