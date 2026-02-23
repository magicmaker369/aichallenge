from openai import OpenAI
import time

YOUR_API_KEY = ""
# Set your actual prices from routerai.ru.
# Example units: price per 1M tokens.
INPUT_PRICE_PER_1M = 174.0
OUTPUT_PRICE_PER_1M = 1396.0
CURRENCY = "RUB"

client = OpenAI(
    api_key=YOUR_API_KEY,
    base_url="https://routerai.ru/api/v1"
)

messages = []

print("Terminal chat started. Type 'exit' to quit.")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Bye!")
        break

    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    started_at = time.perf_counter()
    response = client.chat.completions.create(
        model="openai/gpt-5.2",
        messages=messages
    )
    elapsed_seconds = time.perf_counter() - started_at

    assistant_text = response.choices[0].message.content
    usage = response.usage
    input_cost = (usage.prompt_tokens / 1_000_000) * INPUT_PRICE_PER_1M
    output_cost = (usage.completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    total_cost = input_cost + output_cost

    print(f"AI: {assistant_text}")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Response time: {elapsed_seconds:.2f} sec")
    print(f"Input cost: {input_cost:.6f} {CURRENCY}")
    print(f"Output cost: {output_cost:.6f} {CURRENCY}")
    print(f"Total cost: {total_cost:.6f} {CURRENCY}")

    messages.append({"role": "assistant", "content": assistant_text})
