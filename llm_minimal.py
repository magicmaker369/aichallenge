from openai import OpenAI

YOUR_API_KEY = ""

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

    response = client.chat.completions.create(
        model="openai/gpt-5.2",
        messages=messages
    )

    assistant_text = response.choices[0].message.content
    print(f"AI: {assistant_text}")

    messages.append({"role": "assistant", "content": assistant_text})
