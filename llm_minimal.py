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
DEFAULT_HISTORY_LIMIT = 10

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


def metrics_path(session_id: str) -> Path:
    return HISTORY_DIR / f"session_{session_id}_metrics.json"


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


def default_metrics() -> dict:
    return {
        "response_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "total_response_time_sec": 0.0,
    }


def load_metrics(session_id: str) -> dict:
    path = metrics_path(session_id)
    if not path.exists():
        return default_metrics()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_metrics()

    metrics = default_metrics()
    if not isinstance(data, dict):
        return metrics

    for key, default_value in metrics.items():
        value = data.get(key, default_value)
        if isinstance(default_value, int):
            try:
                metrics[key] = int(value)
            except (TypeError, ValueError):
                metrics[key] = default_value
        else:
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                metrics[key] = default_value

    return metrics


def save_metrics(session_id: str, metrics: dict) -> None:
    path = metrics_path(session_id)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


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


def dialog_messages(messages: list[dict]) -> list[dict]:
    return [item for item in messages if item.get("role") in {"user", "assistant"}]


def print_history(messages: list[dict], limit: Optional[int] = DEFAULT_HISTORY_LIMIT) -> None:
    visible_messages = dialog_messages(messages)
    if not visible_messages:
        print("\nNo previous messages in this session.")
        return

    if limit is None:
        messages_to_show = visible_messages
        print("\nPrevious messages (all):")
    else:
        messages_to_show = visible_messages[-limit:]
        print(f"\nPrevious messages (last {limit}):")

    for item in messages_to_show:
        role = item.get("role")
        role_label = "You" if role == "user" else "AI"
        content = str(item.get("content", "")).strip() or "<empty>"
        print(f"{role_label}: {content}")


def print_summary(messages: list[dict], metrics: dict) -> None:
    visible_messages = dialog_messages(messages)
    user_messages = sum(1 for item in visible_messages if item.get("role") == "user")
    assistant_messages = sum(1 for item in visible_messages if item.get("role") == "assistant")

    response_count = metrics.get("response_count", 0)
    total_response_time_sec = metrics.get("total_response_time_sec", 0.0)
    average_response_time = (
        total_response_time_sec / response_count if response_count > 0 else 0.0
    )

    print("\nSession summary:")
    print(f"Session messages (user+assistant): {len(visible_messages)}")
    print(f"User messages: {user_messages}")
    print(f"Assistant messages: {assistant_messages}")
    print(f"Tracked responses: {response_count}")
    print(f"Prompt tokens (sum): {metrics.get('prompt_tokens', 0)}")
    print(f"Completion tokens (sum): {metrics.get('completion_tokens', 0)}")
    print(f"Total tokens (sum): {metrics.get('total_tokens', 0)}")
    print(f"Input cost (sum): {metrics.get('input_cost', 0.0):.6f} {CURRENCY}")
    print(f"Output cost (sum): {metrics.get('output_cost', 0.0):.6f} {CURRENCY}")
    print(f"Total cost (sum): {metrics.get('total_cost', 0.0):.6f} {CURRENCY}")
    print(f"Average response time: {average_response_time:.2f} sec")


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
session_metrics = load_metrics(current_session_id)

print_history(messages, DEFAULT_HISTORY_LIMIT)
print(
    f"\nTerminal chat started. Session: {current_session_id}. "
    "Type 'exit' to quit. Use '/history all' to print full history. Use '/summary' for session stats."
)

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Bye!")
        break

    if user_input.lower() == "/history all":
        print_history(messages, limit=None)
        continue

    if user_input.lower() == "/history":
        print_history(messages, limit=DEFAULT_HISTORY_LIMIT)
        continue

    if user_input.lower() == "/summary":
        print_summary(messages, session_metrics)
        continue

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

    session_metrics["response_count"] += 1
    session_metrics["prompt_tokens"] += prompt_tokens
    session_metrics["completion_tokens"] += completion_tokens
    session_metrics["total_tokens"] += total_tokens
    session_metrics["input_cost"] += input_cost
    session_metrics["output_cost"] += output_cost
    session_metrics["total_cost"] += total_cost
    session_metrics["total_response_time_sec"] += elapsed_seconds

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
    save_metrics(current_session_id, session_metrics)
