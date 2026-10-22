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

Я не добавляю никаких выходных файлов отчетов, если этого явно не указано в задании.

## architecture principles:

### классы T*
Эти классы может создавать только пользователь
Публичные члены класса - только те, которые задал пользователь
зависимости могут быть любыми, но лучше придерживаться тех пакетов, что упоминаются в сигнатуре класса
Могут наследоваться от Base*, от других T*. Взаиодействовать могут с T*, Base*


### классы Base*
Сущности с Base* могут взаимодействовать только с другими Base* сущностями.
Base* могут быть наследниками только других Base* и типов из стандартной библиотеки python.
Сущности Base* не должны ссылаться на пакеты кроме стандартных питоновских библиотек.
Нельзя создавать новые сущности Base*, не перечисленные пользователем
переменные и методы в ТЗ по сущности Base* - это те и только те члены класса, которые могут использовать другие объекты у любого наследника данного класса.


## Codebase structure:
NEREL_MultiTask_Notebook.ipynb - главный ноутбук проекта
base/ - all Base* classes are here, nothing but Base* here
T_torch/  - all T* classes using torch here
html - all html generating scripts here

## Agent skills

### Issue tracker

Local Markdown в `.scratch/<feature>/`. Перед работой с задачами читай `docs/agents/issue-tracker.md`.

### Triage labels

Пять стандартных статусов. Перед разбором задач читай `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` и `docs/adr/` в корне проекта. Перед изучением кодовой базы читай `docs/agents/domain.md`.
