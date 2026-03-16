import asyncio
import ast
import json
import queue
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Optional

from openai import OpenAI
from vkusvill_mcp_chat import (
    VKUSVILL_CHAT_SENTINEL,
    discover_tools_via_mcp,
    format_exception_for_user,
    new_vkusvill_state,
    process_vkusvill_turn,
    resolve_vkusvill_endpoint,
    run_vkusvill_mcp_chat,
    to_plain_data,
    vkusvill_mcp_availability_error,
    vkusvill_memory_text,
)

YOUR_API_KEY = ""
# Set your actual prices from routerai.ru.
# Example units: price per 1M tokens.
INPUT_PRICE_PER_1M = 174.0
OUTPUT_PRICE_PER_1M = 1396.0
CURRENCY = "RUB"
DEFAULT_HISTORY_LIMIT = 10
STRATEGY_FULL_CONTEXT = 0
STRATEGY_ROLLING_SUMMARY = 1
STRATEGY_SLIDING_WINDOW = 2
STRATEGY_STICKY_FACTS = 3
STRATEGY_LABELS = {
    STRATEGY_FULL_CONTEXT: "Not use strategy (full context)",
    STRATEGY_ROLLING_SUMMARY: "Rolling summary (only last message full context)",
    STRATEGY_SLIDING_WINDOW: "Sliding Window (save only 2 last message context)",
    STRATEGY_STICKY_FACTS: "Sticky Facts (add fact for context)",
}
TOD_PROFILE_DEFAULT = (
    "Ты Tod, шеф-повар по всем существующим кухням мира, "
    "имеешь 5 звезд Мишлен."
)
TOD_OUT_OF_SCOPE_MESSAGE = (
    "У меня нет экспертизы по данному вопросу, вам лучше выбрать чат с другим экспертом"
)
SUMMARY_SYSTEM_PROMPT = (
    "Use the provided conversation summary as context from previous turns. "
    "Treat it as trusted prior chat memory."
)
STICKY_FACTS_SYSTEM_PROMPT = (
    "Use sticky facts as mandatory context from previous turns. "
    "If user request conflicts with sticky facts, ask clarifying question."
)
TOD_SYSTEM_PROMPT_TEMPLATE = (
    "You are Tod chef assistant.\n"
    "Long-term profile:\n{profile}\n\n"
    "Domain limitation:\n"
    "You only help with cooking recipes and food preparation.\n"
    "For any non-cooking question, answer exactly this text and nothing else:\n"
    f"{TOD_OUT_OF_SCOPE_MESSAGE}"
)
PERSONALITY_AGENT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a highly professional assistant.\n"
    "Assistant profile:\n{profile}\n\n"
    "User preferences (style, format, constraints):\n{preferences}\n\n"
    "Follow the profile and preferences strictly in every answer."
)
TRAVEL_PLANNING_SYSTEM_PROMPT = (
    "You are a world-class travel industry expert.\n"
    "Current stage: Planning.\n"
    "Your goal is to gather all requirements, ask clarifying questions, and form a clear plan.\n"
    "Answer in Russian."
)
TRAVEL_PLANNING_CLARIFICATION_PROMPT = (
    "You are a world-class travel industry expert.\n"
    "Current stage: Planning.\n"
    "Ask exactly one clarifying question based on user requirements and current context.\n"
    "Do not generate a plan.\n"
    "Answer in Russian."
)
TRAVEL_PLAN_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a world-class travel industry expert.\n"
    "Create a final approved travel research plan from requirements.\n"
    "Return only the plan in Russian, structured by numbered steps."
)
TRAVEL_EXECUTION_SYSTEM_PROMPT = (
    "You are a world-class travel industry expert.\n"
    "Current stage: Execution.\n"
    "Collect information strictly according to approved plan.\n"
    "Return structured result in Russian with sections by plan steps."
)
TRAVEL_VALIDATION_SYSTEM_PROMPT = (
    "You are a world-class travel industry expert.\n"
    "Current stage: Validation.\n"
    "Validate execution data for relevance, consistency, and match with approved plan.\n"
    "For any item without enough confirmation, append exact warning text:\n"
    "по данной пункту подтверждений найти не удалось\n"
    "Return validated structured result in Russian and include useful links."
)
SUMMARY_UPDATE_SYSTEM_PROMPT = (
    "You update a running summary of a chat. Keep only facts, decisions, constraints, "
    "user preferences, and unresolved questions. Remove repetition. "
    "Write concise plain text in Russian. Return only the updated summary."
)
WEATHER_DEFAULT_CLARIFICATION = (
    "Уточните, для какого города или населённого пункта нужна погода."
)
WEATHER_INTENT_SYSTEM_PROMPT = (
    "You convert weather questions into strict JSON.\n"
    "Return only one JSON object with keys: action, location, days, clarification.\n"
    "Rules:\n"
    "- action must be 'clarify' or 'lookup'\n"
    "- days must be integer from 1 to 3\n"
    "- use 1 for current weather/today, 2 for tomorrow, 3 for day after tomorrow or 3-day forecast\n"
    "- use recent weather chat context if user refers to previous location with short follow-up\n"
    "- if the request is missing a usable location, set action='clarify'\n"
    "- clarification must be a short Russian question\n"
    "- for lookup, location must be a plain string without commentary\n"
    "Do not add markdown or text outside JSON."
)
WEATHER_RESPONSE_SYSTEM_PROMPT = (
    "You are a weather assistant.\n"
    "Answer in Russian.\n"
    "Use only the provided tool JSON, do not invent facts.\n"
    "If status is ambiguous_location, ask the user to choose one of the candidates.\n"
    "If status is not_found, ask the user to specify another location.\n"
    "If status is upstream_error, explain the failure briefly and suggest trying again.\n"
    "If status is ok, summarize the current weather and available forecast briefly.\n"
    "No markdown tables."
)
MULTIMCP_ROUTER_DEFAULT_CLARIFICATION = (
    "Уточните, вам нужна погода или подбор продуктов?"
)
MULTIMCP_ROUTER_SYSTEM_PROMPT = (
    "You route user messages inside a combined MultiMCP chat.\n"
    "Return only one JSON object with keys: action, domain, clarification.\n"
    "Rules:\n"
    "- action must be exactly 'route' or 'clarify'\n"
    "- domain must be exactly 'weather', 'vkusvill', or empty string\n"
    "- clarification must be a short Russian question when action='clarify', otherwise empty\n"
    "- choose 'weather' only for weather forecast, temperature, precipitation, or weather follow-up requests\n"
    "- choose 'vkusvill' only for shopping, grocery, product search, or basket-building requests\n"
    "- if the message mixes both domains or you are not confident, use action='clarify'\n"
    "Do not add markdown or text outside JSON."
)
WEATHER_AUTO_LOCATION = "Москва, Россия"
WEATHER_AUTO_DAYS = 1
WEATHER_AUTO_INTERVAL_SECONDS = 30.0
WEATHER_AUTO_COMMAND_PROMPT = "Command: "
WEATHER_AUTO_RESULTS_DIR = Path("weather_auto_sessions")

HISTORY_DIR = Path("history")
LAST_SESSION_FILE = HISTORY_DIR / "last_session.txt"
TOD_DIR = HISTORY_DIR / "tod_chef_cooking"
TOD_WORKING_MEMORY_DIR = TOD_DIR / "working_memory"
TOD_LONG_TERM_MEMORY_FILE = TOD_DIR / "long_term_memory.json"
TOD_LAST_CHAT_FILE = TOD_DIR / "last_chat.txt"
TOD_CHAT_SENTINEL = "__TOD_CHAT__"
AGENT_CHAT_SENTINEL = "__PERSONALITY_AGENT_CHAT__"
AGENT_PLAN_TRAVEL_CHAT_SENTINEL = "__AGENT_PLAN_TRAVEL_CHAT__"
AGENT_WEATHER_MCP_CHAT_SENTINEL = "__AGENT_WEATHER_MCP_CHAT__"
MULTIMCP_CHAT_SENTINEL = "__MULTIMCP_CHAT__"

MULTIMCP_WEATHER_HINTS = (
    "погода",
    "прогноз",
    "температур",
    "дожд",
    "снег",
    "ветер",
    "градус",
    "пасмур",
    "солнеч",
)
MULTIMCP_WEATHER_FOLLOWUP_HINTS = (
    "сегодня",
    "завтра",
    "послезавтра",
    "сейчас",
    "там",
    "тут",
    "здесь",
)
MULTIMCP_PRODUCT_HINTS = (
    "корзин",
    "товар",
    "продукт",
    "вкусвилл",
    "купи",
    "купить",
    "добавь",
    "добавить",
    "подбери",
    "подобрать",
    "найди",
    "собери",
    "закажи",
)


client = OpenAI(
    api_key=YOUR_API_KEY,
    base_url="https://routerai.ru/api/v1"
)


def ensure_history_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def ensure_weather_auto_results_dir() -> None:
    WEATHER_AUTO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_tod_dirs() -> None:
    TOD_DIR.mkdir(parents=True, exist_ok=True)
    TOD_WORKING_MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def tod_working_memory_path(chat_id: str) -> Path:
    return TOD_WORKING_MEMORY_DIR / f"tod_chat_{chat_id}.json"


def session_path(session_id: str) -> Path:
    return HISTORY_DIR / f"session_{session_id}.json"


def summary_path(session_id: str) -> Path:
    return HISTORY_DIR / f"session_{session_id}_summary.json"


def strategy_path(session_id: str) -> Path:
    return HISTORY_DIR / f"session_{session_id}_strategy.json"


def facts_path(session_id: str) -> Path:
    return HISTORY_DIR / f"session_{session_id}_facts.json"


def metrics_path(session_id: str) -> Path:
    return HISTORY_DIR / f"session_{session_id}_metrics.json"


def create_weather_auto_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def weather_auto_session_path(session_id: str) -> Path:
    return WEATHER_AUTO_RESULTS_DIR / f"session_{session_id}.txt"


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


def default_summary_data() -> dict:
    return {"summary": ""}


def load_summary(session_id: str) -> dict:
    path = summary_path(session_id)
    if not path.exists():
        return default_summary_data()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_summary_data()

    if not isinstance(data, dict):
        return default_summary_data()

    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = ""

    return {"summary": summary}


def save_summary(session_id: str, summary_data: dict) -> None:
    path = summary_path(session_id)
    path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")


def default_facts_data() -> dict:
    return {"facts": []}


def load_facts(session_id: str) -> dict:
    path = facts_path(session_id)
    if not path.exists():
        return default_facts_data()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_facts_data()

    if not isinstance(data, dict):
        return default_facts_data()

    raw_facts = data.get("facts", [])
    if not isinstance(raw_facts, list):
        return default_facts_data()

    facts = [str(item).strip() for item in raw_facts if str(item).strip()]
    return {"facts": facts}


def save_facts(session_id: str, facts_data: dict) -> None:
    path = facts_path(session_id)
    path.write_text(json.dumps(facts_data, ensure_ascii=False, indent=2), encoding="utf-8")


def default_tod_working_memory_data() -> dict:
    return {"facts": []}


def load_tod_working_memory(chat_id: str) -> dict:
    path = tod_working_memory_path(chat_id)
    if not path.exists():
        return default_tod_working_memory_data()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_tod_working_memory_data()

    if not isinstance(data, dict):
        return default_tod_working_memory_data()

    raw_facts = data.get("facts", [])
    if not isinstance(raw_facts, list):
        return default_tod_working_memory_data()

    facts = [str(item).strip() for item in raw_facts if str(item).strip()]
    return {"facts": facts}


def save_tod_working_memory(chat_id: str, memory_data: dict) -> None:
    path = tod_working_memory_path(chat_id)
    path.write_text(json.dumps(memory_data, ensure_ascii=False, indent=2), encoding="utf-8")


def default_tod_long_term_memory_data() -> dict:
    return {"profile": TOD_PROFILE_DEFAULT}


def load_tod_long_term_memory() -> dict:
    ensure_tod_dirs()
    if not TOD_LONG_TERM_MEMORY_FILE.exists():
        data = default_tod_long_term_memory_data()
        TOD_LONG_TERM_MEMORY_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return data

    try:
        data = json.loads(TOD_LONG_TERM_MEMORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = default_tod_long_term_memory_data()
        TOD_LONG_TERM_MEMORY_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return data

    if not isinstance(data, dict):
        return default_tod_long_term_memory_data()

    profile = data.get("profile", TOD_PROFILE_DEFAULT)
    if not isinstance(profile, str) or not profile.strip():
        profile = TOD_PROFILE_DEFAULT

    return {"profile": profile.strip()}


def list_tod_chat_ids() -> list[str]:
    ensure_tod_dirs()
    ids = []
    for path in TOD_WORKING_MEMORY_DIR.glob("tod_chat_*.json"):
        name = path.stem
        if name.startswith("tod_chat_"):
            ids.append(name.replace("tod_chat_", "", 1))
    return sorted(ids, reverse=True)


def create_tod_chat_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_tod_last_chat_id() -> Optional[str]:
    ensure_tod_dirs()
    if not TOD_LAST_CHAT_FILE.exists():
        return None

    chat_id = TOD_LAST_CHAT_FILE.read_text(encoding="utf-8").strip()
    if not chat_id:
        return None

    if not tod_working_memory_path(chat_id).exists():
        return None

    return chat_id


def write_tod_last_chat_id(chat_id: str) -> None:
    ensure_tod_dirs()
    TOD_LAST_CHAT_FILE.write_text(chat_id, encoding="utf-8")


def choose_tod_chat() -> Optional[str]:
    ensure_tod_dirs()
    last_chat_id = read_tod_last_chat_id()

    while True:
        print("\nСhat with Tod chef cooking")
        option_map = {}
        option = 1

        if last_chat_id:
            print(f"{option}) Continue last Tod chat ({last_chat_id})")
            option_map[str(option)] = "continue_last"
            option += 1

        print(f"{option}) Continue existing Tod chat")
        option_map[str(option)] = "continue_existing"
        option += 1

        print(f"{option}) Start new Tod chat")
        option_map[str(option)] = "new"
        option += 1

        print(f"{option}) Back to session menu")
        option_map[str(option)] = "back"

        choice = input("Select option: ").strip()
        action = option_map.get(choice)

        if action == "continue_last" and last_chat_id:
            return last_chat_id

        if action == "continue_existing":
            chat_ids = list_tod_chat_ids()
            if not chat_ids:
                print("No Tod chats yet.")
                continue

            print("\nTod chats:")
            for idx, chat_id in enumerate(chat_ids, start=1):
                memory_data = load_tod_working_memory(chat_id)
                print(f"{idx}) {chat_id} | working memory facts: {len(memory_data.get('facts', []))}")

            picked = input("Select Tod chat number (or press Enter to cancel): ").strip()
            if not picked:
                continue

            if not picked.isdigit():
                print("Invalid input.")
                continue

            index = int(picked) - 1
            if index < 0 or index >= len(chat_ids):
                print("Invalid Tod chat number.")
                continue

            chat_id = chat_ids[index]
            write_tod_last_chat_id(chat_id)
            return chat_id

        if action == "new":
            chat_id = create_tod_chat_id()
            save_tod_working_memory(chat_id, default_tod_working_memory_data())
            write_tod_last_chat_id(chat_id)
            return chat_id

        if action == "back":
            return None

        print("Invalid option. Try again.")


def build_tod_request_messages(
    long_term_memory: dict,
    working_memory: dict,
    short_term_messages: list[dict],
    user_input: str,
) -> list[dict]:
    profile = str(long_term_memory.get("profile", TOD_PROFILE_DEFAULT)).strip() or TOD_PROFILE_DEFAULT
    request_messages = [
        {"role": "system", "content": TOD_SYSTEM_PROMPT_TEMPLATE.format(profile=profile)}
    ]

    facts = working_memory.get("facts", [])
    if facts:
        facts_text = "\n".join(f"- {fact}" for fact in facts)
        request_messages.append(
            {"role": "system", "content": f"Working memory facts:\n{facts_text}"}
        )

    request_messages.extend(short_term_messages)
    request_messages.append({"role": "user", "content": user_input})
    return request_messages


def prompt_and_save_tod_fact(chat_id: str, working_memory: dict, prompt_text: str) -> None:
    fact_input = input(prompt_text).strip()

    if fact_input == "0":
        print("Fact saving skipped.")
        return

    if not fact_input:
        print("Fact was empty, nothing saved.")
        return

    try:
        working_memory["facts"].append(fact_input)
        save_tod_working_memory(chat_id, working_memory)
        print("Fact was saved to working memory.")
    except Exception as error:
        print(f"Failed to save fact: {error}")


def run_tod_chat(chat_id: str) -> str:
    write_tod_last_chat_id(chat_id)
    long_term_memory = load_tod_long_term_memory()
    working_memory = load_tod_working_memory(chat_id)
    short_term_messages = []

    print(
        f"\nСhat with Tod chef cooking started. Tod chat: {chat_id}. "
        "Type 'exit' to quit program, '/switch' to return session menu, "
        "or '/add_in_memory' to save working memory."
    )
    print("In /add_in_memory mode, enter 0 to skip saving.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            return "exit"

        if user_input.lower() == "/switch":
            print("Returning to session menu...")
            return "switch"

        if user_input.lower() == "/add_in_memory":
            prompt_and_save_tod_fact(
                chat_id,
                working_memory,
                "Enter fact to save (enter 0 to skip): ",
            )
            continue

        if not user_input:
            continue

        request_messages = build_tod_request_messages(
            long_term_memory=long_term_memory,
            working_memory=working_memory,
            short_term_messages=short_term_messages,
            user_input=user_input,
        )

        started_at = time.perf_counter()
        response = client.chat.completions.create(
            model="openai/gpt-5.2",
            messages=request_messages,
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

        short_term_messages.append({"role": "user", "content": user_input})
        short_term_messages.append({"role": "assistant", "content": assistant_text})

        prompt_and_save_tod_fact(
            chat_id,
            working_memory,
            "Add fact for next context (enter 0 to skip): ",
        )


def prompt_required_input(prompt_text: str) -> str:
    while True:
        print(prompt_text)
        value = input("> ").strip()
        if value:
            return value
        print("Поле не может быть пустым. Введите текст.")


def run_personality_agent_chat() -> str:
    profile = prompt_required_input("Задайте профиль ассистента")
    preferences = prompt_required_input("Опишите предпочтения (стиль, формат, ограничения)")

    system_prompt = PERSONALITY_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        profile=profile,
        preferences=preferences,
    )
    short_term_messages = []

    print(
        "\nCreate personality agent started. "
        "Type 'exit' to quit program or '/switch' to return session menu."
    )

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            return "exit"

        if user_input.lower() == "/switch":
            print("Returning to session menu...")
            return "switch"

        if not user_input:
            continue

        request_messages = [{"role": "system", "content": system_prompt}]
        request_messages.extend(short_term_messages)
        request_messages.append({"role": "user", "content": user_input})

        started_at = time.perf_counter()
        response = client.chat.completions.create(
            model="openai/gpt-5.2",
            messages=request_messages,
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

        short_term_messages.append({"role": "user", "content": user_input})
        short_term_messages.append({"role": "assistant", "content": assistant_text})


def model_completion_with_stats(request_messages: list[dict]) -> tuple[str, dict]:
    started_at = time.perf_counter()
    response = client.chat.completions.create(
        model="openai/gpt-5.2",
        messages=request_messages,
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

    stats = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed_seconds,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
    return assistant_text, stats


def print_ai_with_stats(assistant_text: str, stats: dict) -> None:
    print(f"AI: {assistant_text}")
    print(f"Prompt tokens: {stats['prompt_tokens']}")
    print(f"Completion tokens: {stats['completion_tokens']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Response time: {stats['elapsed_seconds']:.2f} sec")
    print(f"Input cost: {stats['input_cost']:.6f} {CURRENCY}")
    print(f"Output cost: {stats['output_cost']:.6f} {CURRENCY}")
    print(f"Total cost: {stats['total_cost']:.6f} {CURRENCY}")


def empty_completion_stats() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "elapsed_seconds": 0.0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
    }


def merge_completion_stats(stats_items: list[dict], elapsed_seconds: Optional[float] = None) -> dict:
    merged = empty_completion_stats()
    for item in stats_items:
        if not item:
            continue
        merged["prompt_tokens"] += int(item.get("prompt_tokens", 0))
        merged["completion_tokens"] += int(item.get("completion_tokens", 0))
        merged["total_tokens"] += int(item.get("total_tokens", 0))
        merged["elapsed_seconds"] += float(item.get("elapsed_seconds", 0.0))
        merged["input_cost"] += float(item.get("input_cost", 0.0))
        merged["output_cost"] += float(item.get("output_cost", 0.0))
        merged["total_cost"] += float(item.get("total_cost", 0.0))

    if elapsed_seconds is not None:
        merged["elapsed_seconds"] = elapsed_seconds
    return merged


def parse_json_object(raw_text: str) -> Optional[dict]:
    text = str(raw_text or "").strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if isinstance(payload, dict):
        return payload
    return None


WEATHER_PAYLOAD_KEYS = {
    "status",
    "resolved_location",
    "timezone",
    "current",
    "daily",
    "candidates",
    "message",
}


def parse_relaxed_dict_text(raw_text: str) -> Optional[dict]:
    text = str(raw_text or "").strip()
    if not text:
        return None

    try:
        payload = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None

    if isinstance(payload, dict):
        return payload
    return None


def normalize_weather_payload(value: Any) -> Optional[dict]:
    plain = to_plain_data(value)

    if isinstance(plain, dict):
        if WEATHER_PAYLOAD_KEYS.intersection(plain.keys()):
            return plain

        text_value = plain.get("text")
        if isinstance(text_value, str):
            normalized = normalize_weather_payload(text_value)
            if normalized is not None:
                return normalized

        for key in ("result", "data", "payload", "structuredContent", "structured_content"):
            nested = plain.get(key)
            normalized = normalize_weather_payload(nested)
            if normalized is not None:
                return normalized

    if isinstance(plain, list):
        for item in plain:
            normalized = normalize_weather_payload(item)
            if normalized is not None:
                return normalized
        return None

    if isinstance(plain, str):
        payload = parse_json_object(plain)
        if payload is None:
            payload = parse_relaxed_dict_text(plain)
        if payload is None:
            return None
        return normalize_weather_payload(payload)

    return None


def new_weather_chat_state() -> dict:
    return {
        "messages": [],
        "last_resolved_location": "",
        "last_tool_status": "",
        "last_candidates": [],
    }


def new_multi_mcp_state() -> dict:
    return {
        "messages": [],
        "last_domain": "",
        "pending_domain_clarification": False,
        "weather_state": new_weather_chat_state(),
        "weather_ready": False,
        "vkusvill_state": None,
        "vkusvill_endpoint": "",
        "vkusvill_discovered_tools": [],
        "vkusvill_ready": False,
    }


def append_weather_message(state: dict, role: str, content: str, limit: int = 8) -> None:
    state["messages"].append({"role": role, "content": content})
    if len(state["messages"]) > limit:
        state["messages"] = state["messages"][-limit:]


def append_multi_mcp_message(state: dict, role: str, content: str, limit: int = 10) -> None:
    state["messages"].append({"role": role, "content": content})
    if len(state["messages"]) > limit:
        state["messages"] = state["messages"][-limit:]


def weather_memory_text(state: dict) -> str:
    lines = []

    last_resolved_location = str(state.get("last_resolved_location", "")).strip()
    if last_resolved_location:
        lines.append(f"- Last resolved location: {last_resolved_location}")

    last_tool_status = str(state.get("last_tool_status", "")).strip()
    if last_tool_status:
        lines.append(f"- Last tool status: {last_tool_status}")

    last_candidates = state.get("last_candidates", [])
    if isinstance(last_candidates, list) and last_candidates:
        lines.append("- Recent ambiguous candidates:")
        for idx, label in enumerate(last_candidates, start=1):
            cleaned = str(label).strip()
            if cleaned:
                lines.append(f"  {idx}. {cleaned}")

    if not lines:
        return "(empty)"
    return "\n".join(lines)


def multi_mcp_memory_text(state: dict) -> str:
    lines = []

    last_domain = str(state.get("last_domain", "")).strip()
    if last_domain:
        lines.append(f"- Last routed domain: {last_domain}")

    weather_state = state.get("weather_state")
    if isinstance(weather_state, dict):
        weather_text = weather_memory_text(weather_state)
        if weather_text != "(empty)":
            lines.append("Weather memory:")
            lines.append(weather_text)

    vkusvill_state = state.get("vkusvill_state")
    if isinstance(vkusvill_state, dict):
        vv_text = vkusvill_memory_text(vkusvill_state)
        if vv_text != "(empty)":
            lines.append("VkusVill memory:")
            lines.append(vv_text)

    if not lines:
        return "(empty)"
    return "\n".join(lines)


def build_weather_intent_messages(state: dict, user_input: str) -> list[dict]:
    history_lines = []
    for item in state.get("messages", [])[-6:]:
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        history_lines.append(f"{prefix}: {content}")

    history_text = "\n".join(history_lines) if history_lines else "(empty)"
    memory_text = weather_memory_text(state)
    current_date = datetime.now().strftime("%Y-%m-%d")

    return [
        {"role": "system", "content": WEATHER_INTENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Today: {current_date}\n\n"
                f"Recent weather chat:\n{history_text}\n\n"
                f"Weather memory:\n{memory_text}\n\n"
                f"Latest user message:\n{user_input}\n\n"
                "Return JSON only."
            ),
        },
    ]


def build_multi_mcp_router_messages(state: dict, user_input: str) -> list[dict]:
    history_lines = []
    for item in state.get("messages", [])[-6:]:
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        history_lines.append(f"{prefix}: {content}")

    history_text = "\n".join(history_lines) if history_lines else "(empty)"
    memory_text = multi_mcp_memory_text(state)
    current_date = datetime.now().strftime("%Y-%m-%d")

    return [
        {"role": "system", "content": MULTIMCP_ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Today: {current_date}\n\n"
                f"Recent MultiMCP chat:\n{history_text}\n\n"
                f"Combined memory:\n{memory_text}\n\n"
                f"Latest user message:\n{user_input}\n\n"
                "Return JSON only."
            ),
        },
    ]


def normalize_multi_mcp_route(payload: dict) -> dict:
    action = str(payload.get("action", "")).strip().lower()
    domain = str(payload.get("domain", "")).strip().lower()
    clarification = str(payload.get("clarification", "")).strip()

    if action not in {"route", "clarify"}:
        action = "clarify"

    if domain not in {"weather", "vkusvill"}:
        domain = ""

    if action == "route" and not domain:
        action = "clarify"

    if action == "clarify":
        clarification = clarification or MULTIMCP_ROUTER_DEFAULT_CLARIFICATION
        domain = ""

    return {
        "action": action,
        "domain": domain,
        "clarification": clarification,
    }


def normalize_weather_intent(payload: dict) -> dict:
    action = str(payload.get("action", "")).strip().lower()
    location = payload.get("location")
    clarification = payload.get("clarification")

    try:
        days = int(payload.get("days", 1))
    except (TypeError, ValueError):
        days = 1

    days = max(1, min(3, days))

    if isinstance(location, str):
        location = location.strip() or None
    else:
        location = None

    if isinstance(clarification, str):
        clarification = clarification.strip() or None
    else:
        clarification = None

    if action not in {"clarify", "lookup"}:
        action = "clarify"

    if action == "lookup" and not location:
        action = "clarify"
        clarification = clarification or WEATHER_DEFAULT_CLARIFICATION

    if action == "clarify":
        clarification = clarification or WEATHER_DEFAULT_CLARIFICATION

    return {
        "action": action,
        "location": location,
        "days": days,
        "clarification": clarification,
    }


def normalize_weather_match_text(value: str) -> str:
    normalized = str(value or "").lower().replace("ё", "е")
    normalized = re.sub(r"[^a-zа-я0-9]+", " ", normalized)
    return " ".join(normalized.split())


def clean_weather_location_candidate(value: str) -> Optional[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None

    cleaned = re.sub(
        r"^(?:покажи|скажи|расскажи|дай|хочу\s+узнать|узнай)\s+"
        r"(?:мне\s+)?(?:какая\s+)?(?:сейчас\s+)?(?:погода|прогноз(?:\s+погоды)?)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:какая\s+)?(?:сейчас\s+)?(?:погода|прогноз(?:\s+погоды)?)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:в|во|для|по)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s+(?:на\s+)?(?:сегодня|завтра|послезавтра|сейчас)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:на\s+)?(?:[123]|один|два|три)\s+дн(?:я|ей)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n.,!?;:\"'«»()[]")
    if not cleaned:
        return None

    normalized = normalize_weather_match_text(cleaned)
    if re.fullmatch(r"(?:вариант\s+)?[1-5]", normalized):
        return None
    if re.fullmatch(r"(?:а\s+)?(?:на\s+)?(?:сегодня|завтра|послезавтра|сейчас)", normalized):
        return None
    if normalized in {
        "",
        "pogoda",
        "weather",
        "а сегодня",
        "а завтра",
        "а послезавтра",
        "на сегодня",
        "на завтра",
        "на послезавтра",
        "сегодня",
        "завтра",
        "послезавтра",
        "сейчас",
        "погода",
        "прогноз",
        "прогноз погоды",
    }:
        return None
    return cleaned


def weather_days_from_text(user_input: str) -> int:
    normalized = normalize_weather_match_text(user_input)

    if "послезавтра" in normalized or "через два дня" in normalized:
        return 3
    if re.search(r"\b(?:3|три)\s+дн(?:я|ей)\b", normalized):
        return 3
    if "завтра" in normalized:
        return 2
    if re.search(r"\b(?:2|два)\s+дн(?:я|ей)\b", normalized):
        return 2
    return 1


def resolve_weather_candidate_selection(state: dict, user_input: str) -> Optional[str]:
    raw_candidates = state.get("last_candidates", [])
    candidates = [str(item).strip() for item in raw_candidates if str(item).strip()]
    if not candidates:
        return None

    normalized_input = normalize_weather_match_text(user_input)
    if not normalized_input:
        return None

    ordinal_map = {
        "первый": 1,
        "первая": 1,
        "первое": 1,
        "второй": 2,
        "вторая": 2,
        "второе": 2,
        "третий": 3,
        "третья": 3,
        "третье": 3,
        "четвертый": 4,
        "четвёртый": 4,
        "четвертая": 4,
        "четвёртая": 4,
        "пятый": 5,
        "пятая": 5,
    }

    index_match = re.fullmatch(r"(?:вариант\s+)?([1-5])", normalized_input)
    if index_match:
        index = int(index_match.group(1)) - 1
        if 0 <= index < len(candidates):
            return candidates[index]

    for word, position in ordinal_map.items():
        if normalized_input in {word, f"вариант {word}"}:
            index = position - 1
            if 0 <= index < len(candidates):
                return candidates[index]

    for candidate in candidates:
        normalized_candidate = normalize_weather_match_text(candidate)
        if normalized_input == normalized_candidate:
            return candidate
        if normalized_input and normalized_input in normalized_candidate:
            return candidate

    return None


def extract_weather_location_from_text(state: dict, user_input: str) -> Optional[str]:
    selected_candidate = resolve_weather_candidate_selection(state, user_input)
    if selected_candidate:
        return selected_candidate

    normalized_input = normalize_weather_match_text(user_input)
    patterns = [
        r"\b(?:в|во|для|по)\s+(.+?)(?=(?:\s+(?:на\s+)?(?:сегодня|завтра|послезавтра|сейчас)\b|\s+(?:на\s+)?(?:[123]|один|два|три)\s+дн(?:я|ей)\b|[?!.]|$))",
        r"^(?:какая\s+)?(?:сейчас\s+)?(?:погода|прогноз(?:\s+погоды)?)\s+(.+?)(?=(?:\s+(?:на\s+)?(?:сегодня|завтра|послезавтра|сейчас)\b|\s+(?:на\s+)?(?:[123]|один|два|три)\s+дн(?:я|ей)\b|[?!.]|$))",
        r"^(?:покажи|скажи|расскажи|дай|хочу\s+узнать|узнай)\s+(?:мне\s+)?(?:какая\s+)?(?:сейчас\s+)?(?:погода|прогноз(?:\s+погоды)?)\s+(.+?)(?=(?:\s+(?:на\s+)?(?:сегодня|завтра|послезавтра|сейчас)\b|\s+(?:на\s+)?(?:[123]|один|два|три)\s+дн(?:я|ей)\b|[?!.]|$))",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input, flags=re.IGNORECASE)
        if not match:
            continue
        location = clean_weather_location_candidate(match.group(1))
        if location:
            return location

    last_resolved_location = str(state.get("last_resolved_location", "")).strip()
    if (
        last_resolved_location
        and normalized_input
        and any(
            marker in normalized_input
            for marker in ("сегодня", "завтра", "послезавтра", "сейчас", "погода", "прогноз", "там", "тут", "здесь")
        )
    ):
        return last_resolved_location

    location = clean_weather_location_candidate(user_input)
    if location:
        return location

    return None


def fallback_weather_intent(state: dict, user_input: str) -> dict:
    location = extract_weather_location_from_text(state, user_input)
    if not location:
        return {
            "action": "clarify",
            "location": None,
            "days": 1,
            "clarification": WEATHER_DEFAULT_CLARIFICATION,
        }

    return {
        "action": "lookup",
        "location": location,
        "days": weather_days_from_text(user_input),
        "clarification": WEATHER_DEFAULT_CLARIFICATION,
    }


def extract_weather_intent(state: dict, user_input: str) -> tuple[dict, dict]:
    request_messages = build_weather_intent_messages(state, user_input)
    try:
        assistant_text, stats = model_completion_with_stats(request_messages)
    except Exception:
        return fallback_weather_intent(state, user_input), empty_completion_stats()

    payload = parse_json_object(assistant_text)
    if payload is None:
        return fallback_weather_intent(state, user_input), stats
    return normalize_weather_intent(payload), stats


def multi_mcp_matches_any(text: str, hints: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(hint in lowered for hint in hints)


def extract_multi_mcp_domain_choice(user_input: str) -> Optional[str]:
    lowered = str(user_input or "").strip().lower()
    if not lowered:
        return None

    if any(hint in lowered for hint in ("погод", "weather")):
        return "weather"

    if any(
        hint in lowered
        for hint in (
            "продукт",
            "товар",
            "корзин",
            "вкусвилл",
            "покуп",
            "подбор продуктов",
            "подобрать продукты",
            "продукты",
        )
    ):
        return "vkusvill"

    return None


def looks_like_weather_follow_up(state: dict, user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    if multi_mcp_matches_any(lowered, MULTIMCP_WEATHER_HINTS):
        return True
    if str(state.get("last_domain", "")).strip() != "weather":
        return False
    if multi_mcp_matches_any(lowered, MULTIMCP_WEATHER_FOLLOWUP_HINTS):
        return True

    weather_state = state.get("weather_state")
    if isinstance(weather_state, dict):
        intent = fallback_weather_intent(weather_state, user_input)
        if intent.get("action") == "lookup":
            return True
    return False


def looks_like_product_follow_up(state: dict, user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    if multi_mcp_matches_any(lowered, MULTIMCP_PRODUCT_HINTS):
        return True
    if multi_mcp_matches_any(lowered, MULTIMCP_WEATHER_HINTS):
        return False
    if str(state.get("last_domain", "")).strip() != "vkusvill":
        return False
    return any(marker in lowered for marker in ("добав", "еще", "ещё", "замен", "аналог", "убери", "нужно"))


def fallback_multi_mcp_route(state: dict, user_input: str) -> dict:
    is_weather = looks_like_weather_follow_up(state, user_input)
    is_product = looks_like_product_follow_up(state, user_input)

    if is_weather and is_product:
        return {
            "action": "clarify",
            "domain": "",
            "clarification": MULTIMCP_ROUTER_DEFAULT_CLARIFICATION,
        }
    if is_weather:
        return {"action": "route", "domain": "weather", "clarification": ""}
    if is_product:
        return {"action": "route", "domain": "vkusvill", "clarification": ""}
    return {
        "action": "clarify",
        "domain": "",
        "clarification": MULTIMCP_ROUTER_DEFAULT_CLARIFICATION,
    }


def extract_multi_mcp_route(state: dict, user_input: str) -> tuple[dict, dict]:
    request_messages = build_multi_mcp_router_messages(state, user_input)
    try:
        assistant_text, stats = model_completion_with_stats(request_messages)
    except Exception:
        return fallback_multi_mcp_route(state, user_input), empty_completion_stats()

    payload = parse_json_object(assistant_text)
    if payload is None:
        return fallback_multi_mcp_route(state, user_input), stats
    normalized = normalize_multi_mcp_route(payload)
    if normalized["action"] == "route":
        fallback_route = fallback_multi_mcp_route(state, user_input)
        if fallback_route["action"] == "clarify" and normalized["domain"] not in {"weather", "vkusvill"}:
            return fallback_route, stats
    return normalized, stats


def fallback_weather_response(tool_payload: dict) -> str:
    status = str(tool_payload.get("status", "")).strip()
    message = str(tool_payload.get("message", "")).strip()

    if status == "ambiguous_location":
        labels = []
        for item in tool_payload.get("candidates", []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if label:
                labels.append(label)
        if labels:
            return "Нужно уточнить локацию. Подходящие варианты: " + "; ".join(labels) + "."
        return message or WEATHER_DEFAULT_CLARIFICATION

    if status == "not_found":
        return message or "Не удалось найти такую локацию. Уточните запрос."

    if status == "upstream_error":
        return message or "Не удалось получить данные о погоде. Попробуйте ещё раз."

    resolved_location = tool_payload.get("resolved_location")
    current = tool_payload.get("current")
    daily = tool_payload.get("daily", [])
    timezone = str(tool_payload.get("timezone", "")).strip()

    label = "указанной локации"
    if isinstance(resolved_location, dict):
        resolved_label = str(resolved_location.get("label", "")).strip()
        if resolved_label:
            label = resolved_label

    parts = [f"Погода для {label}."]

    if isinstance(current, dict):
        temperature = current.get("temperature_c")
        apparent = current.get("apparent_temperature_c")
        description = str(current.get("weather_description", "")).strip()
        wind_speed = current.get("wind_speed_kmh")

        current_parts = []
        if temperature is not None:
            current_parts.append(f"Сейчас {temperature}°C")
        if description:
            current_parts.append(description)
        if apparent is not None:
            current_parts.append(f"ощущается как {apparent}°C")
        if wind_speed is not None:
            current_parts.append(f"ветер {wind_speed} км/ч")
        if current_parts:
            parts.append(", ".join(current_parts) + ".")

    if isinstance(daily, list) and daily:
        daily_lines = []
        for item in daily:
            if not isinstance(item, dict):
                continue
            date = str(item.get("date", "")).strip()
            minimum = item.get("temperature_min_c")
            maximum = item.get("temperature_max_c")
            description = str(item.get("weather_description", "")).strip()
            precipitation = item.get("precipitation_sum_mm")

            line_parts = []
            if date:
                line_parts.append(date)
            if minimum is not None and maximum is not None:
                line_parts.append(f"{minimum}..{maximum}°C")
            if description:
                line_parts.append(description)
            if precipitation is not None:
                line_parts.append(f"осадки {precipitation} мм")
            if line_parts:
                daily_lines.append(", ".join(line_parts))
        if daily_lines:
            parts.append("Прогноз: " + "; ".join(daily_lines) + ".")

    if timezone:
        parts.append(f"Часовой пояс: {timezone}.")

    return " ".join(parts).strip()


def summarize_weather_tool_result(user_input: str, tool_payload: dict) -> tuple[str, dict]:
    request_messages = [
        {"role": "system", "content": WEATHER_RESPONSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original user message:\n{user_input}\n\n"
                f"Tool JSON:\n{json.dumps(tool_payload, ensure_ascii=False, indent=2)}\n\n"
                "Write a concise Russian answer."
            ),
        },
    ]

    try:
        assistant_text, stats = model_completion_with_stats(request_messages)
    except Exception:
        return fallback_weather_response(tool_payload), empty_completion_stats()

    assistant_text = assistant_text.strip()
    if not assistant_text:
        return fallback_weather_response(tool_payload), stats
    return assistant_text, stats


def weather_server_path() -> Path:
    return Path(__file__).with_name("weather_mcp_server.py").resolve()


def weather_mcp_availability_error() -> Optional[str]:
    if sys.version_info < (3, 10):
        return "Weather MCP mode requires Python 3.10+."

    server_path = weather_server_path()
    if not server_path.exists():
        return f"Weather MCP server file is missing: {server_path}"

    try:
        from mcp import ClientSession, StdioServerParameters  # noqa: F401
        from mcp.client.stdio import stdio_client  # noqa: F401
    except ImportError:
        return "Weather MCP mode requires the `mcp` package. Install dependencies from README."

    return None


async def _call_weather_tool_via_mcp(location: str, days: int) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(weather_server_path())],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tool_result = await session.call_tool(
                "get_weather",
                arguments={"location": location, "days": days},
            )

    return extract_weather_tool_payload(tool_result)


def unreadable_weather_payload() -> dict:
    return {
        "status": "upstream_error",
        "resolved_location": None,
        "timezone": None,
        "current": None,
        "daily": [],
        "candidates": [],
        "message": "MCP tool returned an unreadable payload.",
    }


def extract_weather_tool_payload(tool_result: Any) -> dict:
    structured = getattr(tool_result, "structuredContent", None)
    if structured is None:
        structured = getattr(tool_result, "structured_content", None)
    if structured is not None:
        normalized = normalize_weather_payload(structured)
        if normalized is not None:
            return normalized

    raw_content = getattr(tool_result, "content", None)
    if raw_content is not None:
        normalized = normalize_weather_payload(raw_content)
        if normalized is not None:
            return normalized

    normalized = normalize_weather_payload(tool_result)
    if normalized is not None:
        return normalized

    return unreadable_weather_payload()


def call_weather_tool_via_mcp(location: str, days: int) -> tuple[dict, float]:
    started_at = time.perf_counter()
    try:
        payload = asyncio.run(_call_weather_tool_via_mcp(location, days))
    except Exception as error:
        payload = {
            "status": "upstream_error",
            "resolved_location": None,
            "timezone": None,
            "current": None,
            "daily": [],
            "candidates": [],
            "message": f"Weather MCP call failed: {error}",
        }
    elapsed_seconds = time.perf_counter() - started_at
    return payload, elapsed_seconds


def update_weather_state_from_tool(state: dict, tool_payload: dict) -> None:
    status = str(tool_payload.get("status", "")).strip()
    state["last_tool_status"] = status

    if status == "ok":
        resolved_location = tool_payload.get("resolved_location")
        if isinstance(resolved_location, dict):
            label = str(resolved_location.get("label", "")).strip()
            if label:
                state["last_resolved_location"] = label
        state["last_candidates"] = []
        return

    if status == "ambiguous_location":
        labels = []
        for item in tool_payload.get("candidates", []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if label:
                labels.append(label)
        state["last_candidates"] = labels
        return

    state["last_candidates"] = []


class WeatherAutoCommandReader:
    def __init__(
        self,
        prompt: str = WEATHER_AUTO_COMMAND_PROMPT,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.prompt = prompt
        self.input_fn = input_fn
        self._commands: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._read_permission = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._read_permission.set()
        self._thread = threading.Thread(
            target=self._run,
            name="weather-auto-command-reader",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._read_permission.wait()
            if self._stop_event.is_set():
                return

            self._read_permission.clear()
            try:
                command = self.input_fn(self.prompt)
            except (EOFError, KeyboardInterrupt):
                command = "exit"
            except Exception:
                command = "exit"

            if self._stop_event.is_set():
                return

            self._commands.put("" if command is None else str(command))

    def get_command(self, timeout: float) -> Optional[str]:
        try:
            return self._commands.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def acknowledge_command(self) -> None:
        if not self._stop_event.is_set():
            self._read_permission.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._read_permission.set()
        if self._thread is None:
            return
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=0.1)


class invariant:
    def __init__(self, max_clarifications: int = 2):
        self.max_clarifications = max_clarifications
        self.clarifications_used = 0
        self.plan_presented = False

    def can_ask_clarification(self) -> bool:
        return self.clarifications_used < self.max_clarifications and not self.plan_presented

    def register_clarification(self) -> None:
        if self.clarifications_used < self.max_clarifications:
            self.clarifications_used += 1

    def clarification_limit_reached(self) -> bool:
        return self.clarifications_used >= self.max_clarifications

    def mark_plan_presented(self) -> None:
        self.plan_presented = True

    def can_approve_plan(self) -> bool:
        return self.plan_presented


def new_travel_plan_state() -> dict:
    return {
        "stage": "planning",
        "planning_notes": [],
        "draft_plan": "",
        "approved_plan": "",
        "execution_output": "",
        "validation_output": "",
        "conversation_messages": [],
    }


def synthesize_travel_plan(planning_notes: list[str], conversation_messages: list[dict]) -> str:
    notes_text = "\n".join(f"- {note}" for note in planning_notes) if planning_notes else "- (empty)"
    convo_text = []
    for item in conversation_messages[-20:]:
        role = item.get("role", "")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        convo_text.append(f"{speaker}: {content}")
    convo_block = "\n".join(convo_text) if convo_text else "(empty)"

    request_messages = [
        {"role": "system", "content": TRAVEL_PLAN_SYNTHESIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Requirements:\n{notes_text}\n\n"
                f"Conversation context:\n{convo_block}\n\n"
                "Create the final approved plan."
            ),
        },
    ]

    plan_text, _stats = model_completion_with_stats(request_messages)
    return plan_text.strip()


def generate_planning_draft_plan(state: dict) -> str:
    return synthesize_travel_plan(
        state["planning_notes"],
        state["conversation_messages"],
    ).strip()


def show_draft_plan(plan_text: str) -> None:
    print("\nGenerated plan:")
    print(plan_text)
    print("\nЕсли нужно изменить план — напишите правки.")
    print("Для перехода в Execution введите /approve_plan.")


def run_travel_execution(plan_text: str, planning_notes: list[str]) -> str:
    notes_text = "\n".join(f"- {note}" for note in planning_notes) if planning_notes else "- (empty)"
    request_messages = [
        {"role": "system", "content": TRAVEL_EXECUTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Approved plan:\n{plan_text}\n\n"
                f"Collected requirements:\n{notes_text}\n\n"
                "Execute plan and collect information."
            ),
        },
    ]
    text, _stats = model_completion_with_stats(request_messages)
    return text.strip()


def run_travel_validation(plan_text: str, execution_output: str) -> str:
    request_messages = [
        {"role": "system", "content": TRAVEL_VALIDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Approved plan:\n{plan_text}\n\n"
                f"Execution output:\n{execution_output}\n\n"
                "Validate and return final validated output."
            ),
        },
    ]
    text, _stats = model_completion_with_stats(request_messages)
    validated = text.strip()

    if "http://" not in validated and "https://" not in validated:
        validated += (
            "\n\nСсылки:\n"
            "- https://www.iatatravelcentre.com\n"
            "- https://www.rome2rio.com\n"
            "- https://www.tripadvisor.com"
        )
    return validated


def run_travel_plan_chat() -> str:
    state = new_travel_plan_state()
    planning_invariant = invariant(max_clarifications=2)

    print(
        "\nAgent with Plan Mode, for travel started. "
        "Pipeline: Planning -> Execution -> Validation -> Done."
    )
    print("Type '/approve_plan' to move from Planning to Execution.")
    print("Type '/switch' to return session menu, or 'exit' to quit program.")

    while True:
        if state["stage"] == "execution":
            print("\nStage: Execution")
            try:
                state["execution_output"] = run_travel_execution(
                    state["approved_plan"],
                    state["planning_notes"],
                )
            except Exception as error:
                print(f"Execution failed: {error}")
                state["stage"] = "planning"
                continue
            state["stage"] = "validation"
            continue

        if state["stage"] == "validation":
            print("\nStage: Validation")
            try:
                state["validation_output"] = run_travel_validation(
                    state["approved_plan"],
                    state["execution_output"],
                )
            except Exception as error:
                print(f"Validation failed: {error}")
                state["stage"] = "planning"
                continue
            print("Сбор информации подготовлен, мне выводить информацию?")
            state["stage"] = "awaiting_done_confirmation"
            continue

        if state["stage"] == "awaiting_done_confirmation":
            user_input = input("You: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Bye!")
                return "exit"

            if user_input.lower() == "/switch":
                print("Returning to session menu...")
                return "switch"

            if user_input.lower() == "yes":
                state["stage"] = "done"
                continue

            if user_input.lower() == "no":
                state["stage"] = "planning"
                continue

            print("Введите Yes или No.")
            continue

        if state["stage"] == "done":
            print("\nStage: Done")
            print(state["validation_output"])
            print("\nDone. You can start a new travel request.")
            state = new_travel_plan_state()
            planning_invariant = invariant(max_clarifications=2)
            continue

        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            return "exit"

        if user_input.lower() == "/switch":
            print("Returning to session menu...")
            return "switch"

        if user_input.lower() == "/approve_plan":
            if not planning_invariant.can_approve_plan():
                print("Команда /approve_plan доступна только после показа плана.")
                continue

            state["approved_plan"] = state["draft_plan"]
            print("\nApproved plan:")
            print(state["approved_plan"])
            print("\nPlan approved. Moving to Execution...")
            state["stage"] = "execution"
            continue

        if not user_input:
            continue

        state["planning_notes"].append(user_input)
        state["conversation_messages"].append({"role": "user", "content": user_input})

        if not planning_invariant.plan_presented:
            if planning_invariant.clarification_limit_reached():
                try:
                    draft_plan = generate_planning_draft_plan(state)
                except Exception as error:
                    print(f"Plan synthesis failed: {error}")
                    continue

                if not draft_plan:
                    print("Не удалось сформировать план. Добавьте правки и попробуйте снова.")
                    continue

                state["draft_plan"] = draft_plan
                planning_invariant.mark_plan_presented()
                show_draft_plan(draft_plan)
                continue

            request_messages = [
                {"role": "system", "content": TRAVEL_PLANNING_SYSTEM_PROMPT},
                {"role": "system", "content": TRAVEL_PLANNING_CLARIFICATION_PROMPT}
            ]
            request_messages.extend(state["conversation_messages"])
            try:
                assistant_text, stats = model_completion_with_stats(request_messages)
            except Exception as error:
                print(f"AI: Не удалось получить ответ: {error}")
                continue

            print_ai_with_stats(assistant_text, stats)
            state["conversation_messages"].append({"role": "assistant", "content": assistant_text})
            planning_invariant.register_clarification()
            continue

        try:
            draft_plan = generate_planning_draft_plan(state)
        except Exception as error:
            print(f"Plan synthesis failed: {error}")
            continue

        if not draft_plan:
            print("Не удалось сформировать план. Добавьте правки и попробуйте снова.")
            continue

        state["draft_plan"] = draft_plan
        show_draft_plan(draft_plan)


def process_weather_turn(
    state: dict,
    user_input: str,
    call_tool_fn: Optional[Callable[[str, int], tuple[dict, float]]] = None,
) -> dict:
    total_started_at = time.perf_counter()
    if call_tool_fn is None:
        call_tool_fn = call_weather_tool_via_mcp

    try:
        weather_intent, intent_stats = extract_weather_intent(state, user_input)
    except Exception as error:
        assistant_text = f"Не удалось обработать запрос о погоде: {error}"
        append_weather_message(state, "user", user_input)
        append_weather_message(state, "assistant", assistant_text)
        state["last_tool_status"] = "error"
        return {
            "assistant_text": assistant_text,
            "stats": empty_completion_stats(),
            "tool_time_seconds": 0.0,
            "tool_summary_line": "",
            "kind": "error",
        }

    if weather_intent["action"] == "clarify":
        assistant_text = weather_intent["clarification"]
        total_stats = merge_completion_stats(
            [intent_stats],
            elapsed_seconds=time.perf_counter() - total_started_at,
        )
        append_weather_message(state, "user", user_input)
        append_weather_message(state, "assistant", assistant_text)
        state["last_tool_status"] = "clarify"
        return {
            "assistant_text": assistant_text,
            "stats": total_stats,
            "tool_time_seconds": 0.0,
            "tool_summary_line": "",
            "kind": "clarify",
        }

    tool_payload, tool_elapsed_seconds = call_tool_fn(
        weather_intent["location"],
        weather_intent["days"],
    )
    update_weather_state_from_tool(state, tool_payload)
    assistant_text, answer_stats = summarize_weather_tool_result(user_input, tool_payload)

    total_stats = merge_completion_stats(
        [intent_stats, answer_stats],
        elapsed_seconds=time.perf_counter() - total_started_at,
    )
    append_weather_message(state, "user", user_input)
    append_weather_message(state, "assistant", assistant_text)
    return {
        "assistant_text": assistant_text,
        "stats": total_stats,
        "tool_time_seconds": tool_elapsed_seconds,
        "tool_summary_line": f"MCP tool time: {tool_elapsed_seconds:.2f} sec",
        "kind": "response",
    }


def run_manual_weather_mcp_chat(state: dict) -> str:
    print(
        "\nWeather via MCP chat started. "
        "Type 'exit' to quit program or '/switch' to return weather menu."
    )

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            return "exit"

        if user_input.lower() == "/switch":
            print("Returning to weather menu...")
            return "switch"

        if not user_input:
            continue

        result = process_weather_turn(state, user_input)
        print_ai_with_stats(result["assistant_text"], result["stats"])
        if result["tool_summary_line"]:
            print(result["tool_summary_line"])


def build_auto_weather_update_text(tool_payload: dict, tool_elapsed_seconds: float) -> str:
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return "\n".join(
        [
            f"[{updated_at}] Автообновление погоды для {WEATHER_AUTO_LOCATION}",
            f"Weather: {fallback_weather_response(tool_payload)}",
            f"MCP tool time: {tool_elapsed_seconds:.2f} sec",
        ]
    )


def print_auto_weather_update(tool_payload: dict, tool_elapsed_seconds: float) -> str:
    update_text = build_auto_weather_update_text(tool_payload, tool_elapsed_seconds)
    print(f"\n{update_text}")
    return update_text


def append_weather_auto_session_entry(session_path: Path, entry_text: str) -> None:
    ensure_weather_auto_results_dir()
    prefix = ""
    if session_path.exists() and session_path.stat().st_size > 0:
        prefix = "\n\n"

    with session_path.open("a", encoding="utf-8") as session_file:
        if prefix:
            session_file.write(prefix)
        session_file.write(entry_text.rstrip())
        session_file.write("\n")


def run_auto_moscow_weather_loop(
    command_reader: Optional[WeatherAutoCommandReader] = None,
    time_fn: Callable[[], float] = time.monotonic,
    interval_seconds: float = WEATHER_AUTO_INTERVAL_SECONDS,
    location: str = WEATHER_AUTO_LOCATION,
    days: int = WEATHER_AUTO_DAYS,
) -> str:
    session_path = weather_auto_session_path(create_weather_auto_session_id())
    print(
        "\nАвтообновление погоды для Москвы запущено. "
        "Используйте '/switch' для возврата в weather menu или 'exit' для выхода."
    )
    print(f"Файл сессии: {session_path}")

    reader = command_reader or WeatherAutoCommandReader()
    reader.start()
    next_update_at = time_fn()

    try:
        while True:
            now = time_fn()
            if now >= next_update_at:
                tool_payload, tool_elapsed_seconds = call_weather_tool_via_mcp(location, days)
                update_text = print_auto_weather_update(tool_payload, tool_elapsed_seconds)
                try:
                    append_weather_auto_session_entry(session_path, update_text)
                    print(f"Информация записана в файл: {session_path}")
                except OSError as error:
                    print(f"Не удалось записать информацию в файл: {error}")
                next_update_at = time_fn() + interval_seconds
                continue

            command = reader.get_command(next_update_at - now)
            if command is None:
                continue

            normalized_command = str(command).strip()
            if not normalized_command:
                reader.acknowledge_command()
                continue

            lowered_command = normalized_command.lower()
            if lowered_command in {"exit", "quit"}:
                print("Bye!")
                return "exit"

            if lowered_command == "/switch":
                print("Returning to weather menu...")
                return "switch"

            print("Unknown command. Use '/switch' to return to weather menu or 'exit' to quit.")
            reader.acknowledge_command()
    finally:
        reader.stop()


def run_weather_mcp_chat() -> str:
    availability_error = weather_mcp_availability_error()
    if availability_error:
        print(f"\nWeather via MCP is unavailable: {availability_error}")
        return "switch"

    state = new_weather_chat_state()

    while True:
        print("\nWeather via MCP")
        print("1) Manual weather request")
        print("2) Присылать данные о погоде на сегодня в Москве каждые 30 секунд")
        print("3) Back to session menu")
        choice = input("Select option: ").strip()

        lowered_choice = choice.lower()
        if lowered_choice in {"exit", "quit"}:
            print("Bye!")
            return "exit"

        if lowered_choice == "/switch" or choice == "3":
            print("Returning to session menu...")
            return "switch"

        if choice == "1":
            result = run_manual_weather_mcp_chat(state)
            if result == "exit":
                return "exit"
            continue

        if choice == "2":
            result = run_auto_moscow_weather_loop()
            if result == "exit":
                return "exit"
            continue

        print("Invalid option. Try again.")


def ensure_multi_mcp_weather_ready(state: dict) -> Optional[str]:
    if state.get("weather_ready"):
        return None

    availability_error = weather_mcp_availability_error()
    if availability_error:
        return availability_error

    state["weather_ready"] = True
    if not isinstance(state.get("weather_state"), dict):
        state["weather_state"] = new_weather_chat_state()
    return None


def ensure_multi_mcp_vkusvill_ready(state: dict) -> Optional[str]:
    if state.get("vkusvill_ready") and isinstance(state.get("vkusvill_state"), dict):
        return None

    availability_error = vkusvill_mcp_availability_error()
    if availability_error:
        return availability_error

    endpoint = str(state.get("vkusvill_endpoint", "")).strip() or resolve_vkusvill_endpoint()
    try:
        discovered_tools, _discovery_elapsed = discover_tools_via_mcp(endpoint)
    except Exception as error:
        return format_exception_for_user(error, endpoint)

    state["vkusvill_endpoint"] = endpoint
    state["vkusvill_discovered_tools"] = discovered_tools
    state["vkusvill_state"] = new_vkusvill_state(
        endpoint=endpoint,
        discovered_tools=discovered_tools,
        persist_history=False,
    )
    state["vkusvill_ready"] = True
    return None


def run_multi_mcp_chat() -> str:
    state = new_multi_mcp_state()

    print(
        "\nMultiMCP chat started. "
        "Type 'exit' to quit program or '/switch' to return session menu."
    )
    print("You can ask about weather or product shopping in one session.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            return "exit"

        if user_input.lower() == "/switch":
            print("Returning to session menu...")
            return "switch"

        if not user_input:
            continue

        total_started_at = time.perf_counter()
        route_stats = empty_completion_stats()
        if state.get("pending_domain_clarification"):
            chosen_domain = extract_multi_mcp_domain_choice(user_input)
            if chosen_domain is not None:
                route_decision = {
                    "action": "route",
                    "domain": chosen_domain,
                    "clarification": "",
                }
                state["pending_domain_clarification"] = False
            else:
                route_decision, route_stats = extract_multi_mcp_route(state, user_input)
        else:
            route_decision, route_stats = extract_multi_mcp_route(state, user_input)

        if route_decision["action"] == "clarify":
            assistant_text = route_decision["clarification"]
            total_stats = merge_completion_stats(
                [route_stats],
                elapsed_seconds=time.perf_counter() - total_started_at,
            )
            print_ai_with_stats(assistant_text, total_stats)
            append_multi_mcp_message(state, "user", user_input)
            append_multi_mcp_message(state, "assistant", assistant_text)
            state["pending_domain_clarification"] = True
            continue

        state["pending_domain_clarification"] = False
        domain = route_decision["domain"]
        if domain == "weather":
            availability_error = ensure_multi_mcp_weather_ready(state)
            if availability_error:
                assistant_text = f"Weather via MCP is unavailable: {availability_error}"
                total_stats = merge_completion_stats(
                    [route_stats],
                    elapsed_seconds=time.perf_counter() - total_started_at,
                )
                print_ai_with_stats(assistant_text, total_stats)
                append_multi_mcp_message(state, "user", user_input)
                append_multi_mcp_message(state, "assistant", assistant_text)
                continue

            result = process_weather_turn(state["weather_state"], user_input)
            total_stats = merge_completion_stats(
                [route_stats, result["stats"]],
                elapsed_seconds=time.perf_counter() - total_started_at,
            )
            print_ai_with_stats(result["assistant_text"], total_stats)
            if result["tool_summary_line"]:
                print(result["tool_summary_line"])

            append_multi_mcp_message(state, "user", user_input)
            append_multi_mcp_message(state, "assistant", result["assistant_text"])
            state["last_domain"] = "weather"
            continue

        availability_error = ensure_multi_mcp_vkusvill_ready(state)
        if availability_error:
            assistant_text = f"VkusVill MCP is unavailable: {availability_error}"
            total_stats = merge_completion_stats(
                [route_stats],
                elapsed_seconds=time.perf_counter() - total_started_at,
            )
            print_ai_with_stats(assistant_text, total_stats)
            append_multi_mcp_message(state, "user", user_input)
            append_multi_mcp_message(state, "assistant", assistant_text)
            continue

        result = process_vkusvill_turn(
            state["vkusvill_state"],
            user_input,
            model_completion_with_stats,
            empty_completion_stats,
            merge_completion_stats,
        )
        total_stats = merge_completion_stats(
            [route_stats, result["stats"]],
            elapsed_seconds=time.perf_counter() - total_started_at,
        )
        print_ai_with_stats(result["assistant_text"], total_stats)
        if result["tool_summary_line"]:
            print(result["tool_summary_line"])

        append_multi_mcp_message(state, "user", user_input)
        append_multi_mcp_message(state, "assistant", result["assistant_text"])
        state["last_domain"] = "vkusvill"


def load_strategy(session_id: str) -> int:
    path = strategy_path(session_id)
    if not path.exists():
        return STRATEGY_ROLLING_SUMMARY

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return STRATEGY_ROLLING_SUMMARY

    if not isinstance(data, dict):
        return STRATEGY_ROLLING_SUMMARY

    value = data.get("strategy", STRATEGY_ROLLING_SUMMARY)
    if value in {
        STRATEGY_FULL_CONTEXT,
        STRATEGY_ROLLING_SUMMARY,
        STRATEGY_SLIDING_WINDOW,
        STRATEGY_STICKY_FACTS,
    }:
        return value
    return STRATEGY_ROLLING_SUMMARY


def save_strategy(session_id: str, strategy: int) -> None:
    path = strategy_path(session_id)
    path.write_text(
        json.dumps({"strategy": strategy}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def choose_context_strategy(session_id: str) -> int:
    current = load_strategy(session_id)
    while True:
        print("\nChoose context management strategies:")
        print('0) "Not use strategy (full context)"')
        print('1) "Rolling summary (only last message full context)"')
        print("2) Sliding Window (save only 2 last message context)")
        print("3) Sticky Facts (add fact for context)")
        choice = input(f"Select option [{current}]: ").strip()

        if not choice:
            strategy = current
            break

        if choice not in {"0", "1", "2", "3"}:
            print("Invalid option. Try again.")
            continue

        strategy = int(choice)
        break

    save_strategy(session_id, strategy)
    return strategy


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
            session_id = name.replace("session_", "", 1)
            if session_id.endswith(("_summary", "_metrics", "_strategy", "_facts")):
                continue
            ids.append(session_id)
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


def trim_messages_for_sliding_window(messages: list[dict], window_size: int = 2) -> list[dict]:
    if window_size <= 0:
        return []
    return list(messages[-window_size:])


def get_last_assistant_message(messages: list[dict]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            return str(item.get("content", "")).strip()
    return ""


def messages_without_last_assistant(messages: list[dict]) -> list[dict]:
    last_assistant_index = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            last_assistant_index = index
            break

    if last_assistant_index is None:
        return list(messages)

    return messages[:last_assistant_index] + messages[last_assistant_index + 1:]


def format_messages_for_summary(messages: list[dict]) -> str:
    lines = []
    for item in messages:
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        prefix = "User" if role == "user" else "Assistant"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def build_request_messages(summary_text: str, last_assistant_text: str, user_input: str) -> list[dict]:
    request_messages = []
    if summary_text.strip():
        request_messages.append(
            {
                "role": "system",
                "content": f"{SUMMARY_SYSTEM_PROMPT}\n\nSummary:\n{summary_text.strip()}",
            }
        )

    if last_assistant_text.strip():
        request_messages.append({"role": "assistant", "content": last_assistant_text.strip()})

    request_messages.append({"role": "user", "content": user_input})
    return request_messages


def build_sticky_facts_request_messages(
    facts: list[str], last_assistant_text: str, user_input: str
) -> list[dict]:
    request_messages = []
    if facts:
        facts_text = "\n".join(f"- {fact}" for fact in facts)
        request_messages.append(
            {
                "role": "system",
                "content": f"{STICKY_FACTS_SYSTEM_PROMPT}\n\nSticky facts:\n{facts_text}",
            }
        )

    if last_assistant_text.strip():
        request_messages.append({"role": "assistant", "content": last_assistant_text.strip()})

    request_messages.append({"role": "user", "content": user_input})
    return request_messages


def update_running_summary(summary_text: str, new_messages: list[dict]) -> str:
    formatted_chunk = format_messages_for_summary(new_messages)
    if not formatted_chunk:
        return summary_text

    summary_request = [
        {"role": "system", "content": SUMMARY_UPDATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Current summary:\n{summary_text.strip() or '(empty)'}\n\n"
                f"New messages:\n{formatted_chunk}\n\n"
                "Return updated summary."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="openai/gpt-5.2",
            messages=summary_request,
        )
    except Exception as error:
        print(f"Warning: summary update failed: {error}")
        return summary_text

    new_summary = response.choices[0].message.content or ""
    return new_summary.strip() or summary_text


def bootstrap_summary_if_needed(session_id: str, messages: list[dict], summary_data: dict) -> dict:
    if summary_data.get("summary", "").strip():
        return summary_data

    source_messages = dialog_messages(messages_without_last_assistant(messages))
    if not source_messages:
        return summary_data

    print("Preparing session summary from previous history...")
    summary_data["summary"] = update_running_summary("", source_messages)
    save_summary(session_id, summary_data)
    return summary_data


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
        print("\nChoose variant for chat session (it's base variant for chating):")
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

        print("- - -")
        print("Сhat with Tod chef cooking (type Tod to open this chat).")
        print("- - -")
        print("Create personality agent (type agent to starting this chat).")
        print("- - -")
        print("Agent with Plan Mode, for travel (type \"agent with plan\" to starting this chat)")
        print("- - -")
        print("Weather via MCP (type \"weather mcp\" to starting this chat)")
        print("- - -")
        print("VkusVill via MCP (type \"vkusvill mcp\" to starting this chat)")
        print("- - -")
        print("MultiMCP (type \"MM\" to starting this chat)")
        print("- - -")

        choice = input("Select option: ").strip()
        if choice.lower() == "tod":
            return TOD_CHAT_SENTINEL, []
        if choice.lower() == "agent":
            return AGENT_CHAT_SENTINEL, []
        if choice.lower() == "agent with plan":
            return AGENT_PLAN_TRAVEL_CHAT_SENTINEL, []
        if choice.lower() == "weather mcp":
            return AGENT_WEATHER_MCP_CHAT_SENTINEL, []
        if choice.lower() == "vkusvill mcp":
            return VKUSVILL_CHAT_SENTINEL, []
        if choice.lower() == "mm":
            return MULTIMCP_CHAT_SENTINEL, []
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


def main() -> None:
    while True:
        current_session_id, messages = choose_session()

        if current_session_id == TOD_CHAT_SENTINEL:
            tod_chat_id = choose_tod_chat()
            if tod_chat_id is None:
                continue

            tod_result = run_tod_chat(tod_chat_id)
            if tod_result == "exit":
                break
            continue

        if current_session_id == AGENT_CHAT_SENTINEL:
            agent_result = run_personality_agent_chat()
            if agent_result == "exit":
                break
            continue

        if current_session_id == AGENT_PLAN_TRAVEL_CHAT_SENTINEL:
            travel_result = run_travel_plan_chat()
            if travel_result == "exit":
                break
            continue

        if current_session_id == AGENT_WEATHER_MCP_CHAT_SENTINEL:
            weather_result = run_weather_mcp_chat()
            if weather_result == "exit":
                break
            continue

        if current_session_id == VKUSVILL_CHAT_SENTINEL:
            vkusvill_result = run_vkusvill_mcp_chat(
                model_completion_with_stats,
                print_ai_with_stats,
                empty_completion_stats,
                merge_completion_stats,
            )
            if vkusvill_result == "exit":
                break
            continue

        if current_session_id == MULTIMCP_CHAT_SENTINEL:
            multi_result = run_multi_mcp_chat()
            if multi_result == "exit":
                break
            continue

        write_last_session_id(current_session_id)
        context_strategy = choose_context_strategy(current_session_id)
        session_metrics = load_metrics(current_session_id)
        session_summary = load_summary(current_session_id)
        session_facts = load_facts(current_session_id)
        if context_strategy == STRATEGY_ROLLING_SUMMARY:
            session_summary = bootstrap_summary_if_needed(current_session_id, messages, session_summary)
        elif context_strategy == STRATEGY_SLIDING_WINDOW:
            messages = trim_messages_for_sliding_window(messages, 2)
            save_session(current_session_id, messages)

        print_history(messages, DEFAULT_HISTORY_LIMIT)
        print(
            f"\nTerminal chat started. Session: {current_session_id}. "
            f"Strategy: {STRATEGY_LABELS[context_strategy]}. "
            "Type 'exit' to quit. Use '/switch' to open session menu. "
            "Use '/history all' to print full history. Use '/summary' for session stats."
        )

        switch_requested = False

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Bye!")
                break

            if user_input.lower() == "/switch":
                print("Returning to session menu...")
                switch_requested = True
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

            last_assistant_text = ""
            if context_strategy == STRATEGY_ROLLING_SUMMARY:
                last_assistant_text = get_last_assistant_message(messages)
                request_messages = build_request_messages(
                    session_summary.get("summary", ""),
                    last_assistant_text,
                    user_input,
                )
            elif context_strategy == STRATEGY_SLIDING_WINDOW:
                request_messages = trim_messages_for_sliding_window(messages, 2)
                request_messages.append({"role": "user", "content": user_input})
            elif context_strategy == STRATEGY_STICKY_FACTS:
                last_assistant_text = get_last_assistant_message(messages)
                request_messages = build_sticky_facts_request_messages(
                    session_facts.get("facts", []),
                    last_assistant_text,
                    user_input,
                )
            else:
                request_messages = list(messages)
                request_messages.append({"role": "user", "content": user_input})

            messages.append({"role": "user", "content": user_input})
            if context_strategy == STRATEGY_SLIDING_WINDOW:
                messages = trim_messages_for_sliding_window(messages, 2)
            save_session(current_session_id, messages)

            started_at = time.perf_counter()
            response = client.chat.completions.create(
                model="openai/gpt-5.2",
                messages=request_messages
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

            if context_strategy == STRATEGY_ROLLING_SUMMARY:
                summary_chunk = []
                if last_assistant_text:
                    summary_chunk.append({"role": "assistant", "content": last_assistant_text})
                summary_chunk.append({"role": "user", "content": user_input})
                session_summary["summary"] = update_running_summary(
                    session_summary.get("summary", ""),
                    summary_chunk,
                )
                save_summary(current_session_id, session_summary)

            messages.append({"role": "assistant", "content": assistant_text})
            if context_strategy == STRATEGY_SLIDING_WINDOW:
                messages = trim_messages_for_sliding_window(messages, 2)
            save_session(current_session_id, messages)
            save_metrics(current_session_id, session_metrics)

            if context_strategy == STRATEGY_STICKY_FACTS:
                while True:
                    fact_input = input(
                        "Add fact for next context (enter 0 to skip): "
                    ).strip()
                    if fact_input == "0":
                        break

                    if not fact_input:
                        print("Please enter a fact or 0 to skip.")
                        continue

                    session_facts["facts"].append(fact_input)
                    save_facts(current_session_id, session_facts)
                    print("Fact saved.")
                    break

        if not switch_requested:
            break


if __name__ == "__main__":
    main()
