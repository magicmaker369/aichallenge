# Terminal LLM Chat (Python)

## Установка
```bash
python3 -m pip install openai
```

## Настройка
Откройте `llm_minimal.py` и вставьте ваш API-ключ в переменную:

```python
YOUR_API_KEY = "YOUR_API_KEY"
```

## Запуск
```bash
python3 llm_minimal.py
```

## Использование
- Пишите вопрос после `You:`
- Модель отвечает строкой `AI: ...`
- Для выхода используйте `exit` или `quit`
- Для вывода общей информации по объему токенов, потраченых средств в рамках чата, используйте команду `/summary`

