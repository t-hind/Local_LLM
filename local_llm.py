"""Private, offline text generation with a local GGUF model."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly and concisely. "
    "Do not claim to have accessed the internet."
)


def build_prompt(system_prompt: str, user_prompt: str) -> str:
    """Build a broadly compatible instruction prompt for an instruct model."""
    return (
        f"<|system|>\n{system_prompt.strip()}\n"
        f"<|user|>\n{user_prompt.strip()}\n"
        "<|assistant|>\n"
    )


class LocalModel:
    """Lazy wrapper around llama.cpp so health checks do not load the model."""

    def __init__(self, model_path: Path, context_size: int, threads: int) -> None:
        self.model_path = model_path
        self.context_size = context_size
        self.threads = threads
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}. "
                    "Place a local .gguf file there or set LOCAL_LLM_MODEL."
                )
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Run: "
                    "python -m pip install -r requirements.txt"
                ) from error

            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                n_threads=self.threads,
                verbose=False,
            )
        return self._model

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        if not user_prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        result = self._load()(
            build_prompt(system_prompt, user_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|user|>", "<|system|>"],
        )
        return str(result["choices"][0]["text"]).strip()


def make_handler(model: LocalModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalLLM/1.0"

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(
                200,
                {"status": "ok", "model": str(model.model_path), "loaded": model._model is not None},
            )

        def do_POST(self) -> None:
            if self.path != "/generate":
                self._send_json(404, {"error": "not found"})
                return
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                request = json.loads(body)
                answer = model.generate(
                    user_prompt=str(request.get("prompt", "")),
                    system_prompt=str(request.get("system", DEFAULT_SYSTEM_PROMPT)),
                    max_tokens=int(request.get("max_tokens", 256)),
                    temperature=float(request.get("temperature", 0.7)),
                )
                self._send_json(200, {"response": answer})
            except (ValueError, TypeError, json.JSONDecodeError, FileNotFoundError, RuntimeError) as error:
                self._send_json(400, {"error": str(error)})

        def log_message(self, format: str, *args: object) -> None:
            print(f"[http] {format % args}")

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            response = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a private local LLM server or one-shot prompt.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("LOCAL_LLM_MODEL", "models/Qwen3-8B-Q4_K_M.gguf")),
    )
    parser.add_argument("--host", default=os.getenv("LOCAL_LLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCAL_LLM_PORT", "8000")))
    parser.add_argument("--context-size", type=int, default=int(os.getenv("LOCAL_LLM_CONTEXT", "4096")))
    parser.add_argument("--threads", type=int, default=int(os.getenv("LOCAL_LLM_THREADS", "4")))
    parser.add_argument("--prompt", help="Generate one response and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = LocalModel(args.model, args.context_size, args.threads)
    if args.prompt:
        print(model.generate(args.prompt))
        return

    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    print(f"Local LLM listening at http://{args.host}:{args.port}")
    print(f"Model: {args.model}")
    print("No external network requests are made by this application.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local LLM.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()