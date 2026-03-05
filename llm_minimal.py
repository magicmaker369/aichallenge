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

HISTORY_DIR = Path("history")
LAST_SESSION_FILE = HISTORY_DIR / "last_session.txt"
TOD_DIR = HISTORY_DIR / "tod_chef_cooking"
TOD_WORKING_MEMORY_DIR = TOD_DIR / "working_memory"
TOD_LONG_TERM_MEMORY_FILE = TOD_DIR / "long_term_memory.json"
TOD_LAST_CHAT_FILE = TOD_DIR / "last_chat.txt"
TOD_CHAT_SENTINEL = "__TOD_CHAT__"
AGENT_CHAT_SENTINEL = "__PERSONALITY_AGENT_CHAT__"
AGENT_PLAN_TRAVEL_CHAT_SENTINEL = "__AGENT_PLAN_TRAVEL_CHAT__"


client = OpenAI(
    api_key=YOUR_API_KEY,
    base_url="https://routerai.ru/api/v1"
)


def ensure_history_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


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
            state["stage"] = "done"
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

        choice = input("Select option: ").strip()
        if choice.lower() == "tod":
            return TOD_CHAT_SENTINEL, []
        if choice.lower() == "agent":
            return AGENT_CHAT_SENTINEL, []
        if choice.lower() == "agent with plan":
            return AGENT_PLAN_TRAVEL_CHAT_SENTINEL, []
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
