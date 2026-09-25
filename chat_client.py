"""Command-line client for the local LLM HTTP server."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class Conversation:
    """Keep conversation history in the client and send it with each request."""

    system_prompt: str
    turns: list[tuple[str, str]] = field(default_factory=list)

    def prompt_for(self, user_message: str) -> str:
        history = [f"System: {self.system_prompt}"]
        for role, message in self.turns:
            history.append(f"{role}: {message}")
        history.append(f"User: {user_message}")
        history.append("Assistant:")
        return "\n\n".join(history)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self.turns.extend([("User", user_message), ("Assistant", assistant_message)])


def send_message(server_url: str, prompt: str, max_tokens: int, temperature: float) -> str:
    payload = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    ).encode("utf-8")
    request = Request(
        f"{server_url.rstrip('/')}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(
            f"Cannot connect to {server_url}. Start the server with: python local_llm.py"
        ) from error

    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return str(result["response"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the local LLM.")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("message", nargs="?", help="Send this message and exit.")
    parser.add_argument("--prompt", help="Send one message and exit.")
    parser.add_argument("--system", default="You are a helpful assistant. /no_think")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conversation = Conversation(args.system)

    one_shot_message = args.prompt or args.message
    if one_shot_message:
        try:
            prompt = conversation.prompt_for(one_shot_message)
            print(send_message(args.server, prompt, args.max_tokens, args.temperature))
        except RuntimeError as error:
            print(f"Error: {error}")
        return

    print("Local LLM chat. Type /exit to quit or /clear to clear the conversation.")
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_message:
            continue
        if user_message.lower() in {"/exit", "/quit"}:
            return
        if user_message.lower() == "/clear":
            conversation.turns.clear()
            print("Conversation cleared.")
            continue

        try:
            prompt = conversation.prompt_for(user_message)
            answer = send_message(args.server, prompt, args.max_tokens, args.temperature)
        except RuntimeError as error:
            print(f"Error: {error}")
            continue
        conversation.add_turn(user_message, answer)
        print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()