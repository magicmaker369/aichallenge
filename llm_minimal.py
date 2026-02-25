import json
from datetime import datetime
from pathlib import Path
import time
from typing import Optional

from openai import OpenAI

YOUR_API_KEY = ""
# Set your actual prices from routerai.ru.
# Example units: price per 1M tokens.
INPUT_PRICE_PER_1M = 174.0
OUTPUT_PRICE_PER_1M = 1396.0
CURRENCY = "RUB"

HISTORY_DIR = Path("history")
LAST_SESSION_FILE = HISTORY_DIR / "last_session.txt"


client = OpenAI(
    api_key=YOUR_API_KEY,
    base_url="https://routerai.ru/api/v1"
)


def ensure_history_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def session_path(session_id: str) -> Path:
    return HISTORY_DIR / f"session_{session_id}.json"


def create_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_session(session_id: str, messages: list[dict]) -> None:
    path = session_path(session_id)
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(session_id: str) -> list[dict]:
    path = session_path(session_id)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return data
    return []


def read_last_session_id() -> Optional[str]:
    if not LAST_SESSION_FILE.exists():
        return None

    session_id = LAST_SESSION_FILE.read_text(encoding="utf-8").strip()
    if not session_id:
        return None

    if not session_path(session_id).exists():
        return None

    return session_id


def write_last_session_id(session_id: str) -> None:
    LAST_SESSION_FILE.write_text(session_id, encoding="utf-8")


def list_session_ids() -> list[str]:
    ids = []
    for path in HISTORY_DIR.glob("session_*.json"):
        name = path.stem
        if name.startswith("session_"):
            ids.append(name.replace("session_", "", 1))
    return sorted(ids, reverse=True)


def session_preview(messages: list[dict]) -> str:
    if not messages:
        return "empty"

    for item in reversed(messages):
        content = str(item.get("content", "")).replace("\n", " ").strip()
        if content:
            return content[:60]

    return "empty"


def choose_session() -> tuple[str, list[dict]]:
    ensure_history_dir()
    last_session_id = read_last_session_id()

    while True:
        print("\nChoose chat session:")
        option_map = {}
        option = 1

        if last_session_id:
            print(f"{option}) Continue last ({last_session_id})")
            option_map[str(option)] = "continue_last"
            option += 1

        print(f"{option}) List sessions")
        option_map[str(option)] = "list"
        option += 1

        print(f"{option}) New session")
        option_map[str(option)] = "new"

        choice = input("Select option: ").strip()
        action = option_map.get(choice)

        if action == "continue_last" and last_session_id:
            messages = load_session(last_session_id)
            return last_session_id, messages

        if action == "list":
            session_ids = list_session_ids()
            if not session_ids:
                print("No saved sessions yet.")
                continue

            print("\nSaved sessions:")
            for idx, session_id in enumerate(session_ids, start=1):
                messages = load_session(session_id)
                preview = session_preview(messages)
                print(f"{idx}) {session_id} | messages: {len(messages)} | {preview}")

            picked = input("Select session number (or press Enter to cancel): ").strip()
            if not picked:
                continue

            if not picked.isdigit():
                print("Invalid input.")
                continue

            index = int(picked) - 1
            if index < 0 or index >= len(session_ids):
                print("Invalid session number.")
                continue

            session_id = session_ids[index]
            messages = load_session(session_id)
            write_last_session_id(session_id)
            return session_id, messages

        if action == "new":
            session_id = create_session_id()
            messages = []
            save_session(session_id, messages)
            write_last_session_id(session_id)
            return session_id, messages

        print("Invalid option. Try again.")


current_session_id, messages = choose_session()
write_last_session_id(current_session_id)

print(f"\nTerminal chat started. Session: {current_session_id}. Type 'exit' to quit.")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Bye!")
        break

    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})
    save_session(current_session_id, messages)

    started_at = time.perf_counter()
    response = client.chat.completions.create(
        model="openai/gpt-5.2",
        messages=messages
    )
    elapsed_seconds = time.perf_counter() - started_at

    assistant_text = response.choices[0].message.content or ""
    usage = response.usage

    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

    input_cost = (prompt_tokens / 1_000_000) * INPUT_PRICE_PER_1M
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    total_cost = input_cost + output_cost

    print(f"AI: {assistant_text}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Response time: {elapsed_seconds:.2f} sec")
    print(f"Input cost: {input_cost:.6f} {CURRENCY}")
    print(f"Output cost: {output_cost:.6f} {CURRENCY}")
    print(f"Total cost: {total_cost:.6f} {CURRENCY}")

    messages.append({"role": "assistant", "content": assistant_text})
    save_session(current_session_id, messages)
