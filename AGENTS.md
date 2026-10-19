## Dev environment
use python environment from here
`/Users/dima/Practikum/nlp-2026/sprint_1/bin`

## dev tools 
I check all the python code with tools:
- pyright mcp server - for static typing
- ruff - for code format
untill it passes

Я не добавляю никаких асертов и проверок правильности финального ответа, правильности выполнения кодового задания
без прямого указания проверки от пользователя

## architecture principles:
Сущности с Base* могут взаимодействовать только с другими Base* сущностями.
Сущности Base* не должны ссылаться на пакеты кроме стандартных питоновских библиотек.
Нельзя создавать новые сущности Base*, не перечисленные пользователем
переменные и методы в ТЗ по сущности Base* - это те и только те члены класса, которые могут использовать другие объекты у любого наследника данного класса.

## Agent skills

### Issue tracker

Local Markdown в `.scratch/<feature>/`. Перед работой с задачами читай `docs/agents/issue-tracker.md`.

### Triage labels

Пять стандартных статусов. Перед разбором задач читай `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` и `docs/adr/` в корне проекта. Перед изучением кодовой базы читай `docs/agents/domain.md`.
