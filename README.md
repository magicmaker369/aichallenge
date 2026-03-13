# Terminal LLM Chat (Python)

## Требования
- Python `3.10+`

## Установка
```bash
python3 -m pip install openai "mcp[cli]" httpx
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

Если хотите запускать проект из локального virtualenv внутри репозитория:
```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install openai "mcp[cli]" httpx
.venv/bin/python llm_minimal.py
```

Запуск одной командой:
```bash
./run.sh
```

## Использование
- Пишите вопрос после `You:`
- Модель отвечает строкой `AI: ...`
- Для выхода используйте `exit` или `quit`
- Для возврата в меню используйте `/switch`

## Специальные режимы
- `Tod` - отдельный чат для кулинарных запросов
- `agent` - создание personality agent
- `agent with plan` - travel-режим с Planning -> Execution -> Validation
- `weather mcp` - отдельный weather-чат с запросом погоды через локальный безопасный MCP server
- `vkusvill mcp` - отдельный чат для подключения к удалённому MCP server ВкусВилл

## Weather via MCP
- Режим `weather mcp` использует локальный read-only MCP server из [`weather_mcp_server.py`](./weather_mcp_server.py)
- Источник погодных данных: Open-Meteo, без API-ключа
- После входа в `weather mcp` появляется подменю:
  - `1` - ручной weather-чат с вводом своих запросов
  - `2` - автообновление погоды на сегодня для `Москва, Россия` каждые 30 секунд
  - `3` - возврат в общее меню приложения
- Запросы погоды проходят через app-managed orchestration:
  - первый вызов LLM превращает сообщение пользователя в структуру `action/location/days`
  - приложение вызывает MCP tool `get_weather`
  - второй вызов LLM формирует короткий ответ на русском
  - если LLM недоступна или возвращает ошибку авторизации, приложение переключается на локальный fallback для разбора запроса и шаблонный ответ
- В режиме `2` LLM не используется: приложение каждые 30 секунд само вызывает MCP tool для Москвы и печатает локально сформированный ответ
- Результаты режима `2` дополнительно сохраняются в отдельную папку проекта `weather_auto_sessions/`
- Для каждого запуска пункта `2` создаётся свой файл сессии; запись идёт после каждого показанного обновления и о ней печатается отдельное информационное сообщение
- Команда `/switch` внутри режимов `1` и `2` возвращает в weather-подменю, `exit` завершает приложение
- Если `mcp` не установлен или версия Python ниже `3.10`, обычные режимы останутся доступны, а `weather mcp` сообщит о недостающей среде

## VkusVill via MCP
- Режим `vkusvill mcp` использует отдельный изолированный контур из [`vkusvill_mcp_chat.py`](./vkusvill_mcp_chat.py)
- По умолчанию используется endpoint `https://mcp001.vkusvill.ru/mcp`
- Для override endpoint задайте переменную окружения `VKUSVILL_MCP_URL`
- История режима хранится отдельно в `history/vkusvill_mcp/`
- Внутри режима доступны команды:
  - `/tools` - показать discovery snapshot доступных MCP tools
  - `/call <tool> <json>` - вручную вызвать tool с JSON-аргументами
  - `/reset` - сбросить локальное состояние VkusVill-сессии
- Если `mcp` не установлен или удалённый endpoint недоступен, остальные режимы продолжат работать как обычно
