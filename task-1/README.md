# Задание 1: RAG по ТФКП

ФИО: Кучеров Константин Вадимович

Группа: М05-607г

Решение по [условию задания](materials/homework_01_rag.md) и [15 выданным QA](materials/tfkp_15_qa.md).

## Структура

```text
task-1/
├── README.md                  # запуск и навигация
├── report.md                  # результаты и разбор ошибок
├── build_index.py             # сборка индекса
├── predict.py                 # ответы на вопросы
├── run_local.py               # запуск на Mac одной командой
├── config.json                # настройки поиска и пути
├── runtime.json               # модель, sampling и лимиты
├── requirements.txt           # основные зависимости
├── dev_answers.jsonl          # сохранённые ответы модели
├── dev_trace.jsonl            # расход токенов и источники
├── experiments.csv            # сравнение вариантов
├── own_questions.jsonl        # авторские вопросы
├── own_gold.jsonl             # эталоны для авторских вопросов
├── notebooks/
│   └── solution.ipynb         # просмотр результатов и запуск
├── datasets/dev/              # открытые вопросы и эталоны
├── materials/                 # условие, исходные QA, пример API
├── rag/                       # поиск, ранжирование, формулы
├── scripts/                   # подготовка, сервер, эксперименты
├── tests/                     # проверки контрактов
├── requirements/              # зависимости Mac и notebook
├── results/                   # подробные метрики
└── local/                     # локальные файлы, исключены из Git
    ├── assets/                # токенизатор
    ├── data/                  # PDF, корпус и журнал извлечения
    ├── models/                # скачанные веса
    ├── index/                 # собранный индекс
    ├── outputs/               # новые ответы и лог сервера
    └── runs/                  # повторные эксперименты
```

## Окружение

Нужен Python 3.11. Следующие команды выполняются из корня репозитория; после `cd` все дальнейшие команды — из `task-1/`.

```bash
cd task-1
python3.11 -m venv ../.venv
source ../.venv/bin/activate
```

Выберите один из двух вариантов запуска ниже. Версии прямых и транзитивных зависимостей зафиксированы в `requirements.txt`; локальная генерация и ноутбук имеют дополнительные файлы зависимостей.

## Подготовка и запуск с готовым API

Для внешнего сервера модели достаточно основных зависимостей. До сборки индекса нужно скачать токенизатор Qwen и обе модели поиска. Команда `prepare all` также скачивает учебник, восстанавливает корпус и создаёт вопросы с эталонами.

```bash
python -m pip install -r requirements.txt
python -m scripts.prepare all --config config.json
python build_index.py --corpus local/data/corpus.jsonl --output local/index --config config.json
```

Укажите OpenAI-совместимый endpoint и имя **заданной Qwen3.5-4B** на сервере. Адрес ниже подходит для локального сервера из следующего раздела; для другого сервера замените его на выданный адрес с окончанием `/v1`.

```bash
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="Qwen/Qwen3.5-4B"
export SUBMISSION_ID="tfkp-dev-run-001"
# Только если endpoint требует ключ: export LLM_API_KEY="..."

python predict.py --index local/index --questions datasets/dev/dev_questions.jsonl --output local/outputs/answers.jsonl --trace local/outputs/trace.jsonl --config config.json
```

`predict.py` использует итоговую конфигурацию. [Пример запроса](materials/example_request.json) показывает формат API. Каждый запрос содержит заголовки `X-Question-ID` и `X-Submission-ID`. Расход токенов берётся из ответа сервера; автоматических повторов запросов нет.

Сеть нужна на этапе установки зависимостей и `prepare assets`/`prepare all`. Сборка индекса работает с уже скачанными файлами. При предсказании сетевые запросы идут только к заданному endpoint; модели поиска загружаются с диска и работают на CPU.

## Полностью локальный запуск на Apple Silicon

После создания и активации окружения установите дополнительные зависимости MLX и скачайте также веса Qwen:

```bash
python -m pip install -r requirements/local.txt
python -m scripts.prepare all --model --config config.json
python build_index.py --corpus local/data/corpus.jsonl --output local/index --config config.json
python run_local.py
```

`run_local.py` запускает сервер Qwen на GPU, обрабатывает открытый набор и останавливает созданный им сервер. Результаты появляются в `local/outputs/answers.jsonl`, `local/outputs/trace.jsonl` и `local/outputs/server.log`. Для своих вопросов передайте `--questions /path/to/questions.jsonl --output-dir /path/to/output`; для индекса в другой папке — `--index /path/to/index`. Этот скрипт сам назначает `SUBMISSION_ID` и адрес локального endpoint.

Локальный сервер использует MLX и требует Apple Silicon с доступом к Metal. На машине без этого backend используйте вариант с готовым API. Для просмотра и запуска ноутбука дополнительно установите `python -m pip install -r requirements/notebook.txt` и выберите это окружение в Jupyter или VS Code.

## Другие пути и восстановление данных

При запуске из другой папки передайте абсолютные пути к скриптам, конфигурации и входным файлам:

```bash
python /path/to/repository/task-1/build_index.py --corpus /path/to/corpus.jsonl --output /path/to/index --config /path/to/repository/task-1/config.json
python /path/to/repository/task-1/predict.py --index /path/to/index --questions /path/to/questions.jsonl --output /path/to/answers.jsonl --trace /path/to/trace.jsonl --config /path/to/repository/task-1/config.json
```

Пути к скачанным моделям в `config.json` вычисляются относительно самой конфигурации. Все файлы готового индекса, включая модели поиска и токенизатор, записываются в `--output`.

`build_index.py` создаёт индекс формата 3 с нуля: поисковые окна, векторы, оглавление, указатели названий теорем и структурных признаков формул выводятся только из корпуса. Сборка не читает вопросы и эталоны, не вызывает LLM и не требует сохранённого индекса. После обновления кода индексы форматов 1 и 2 нужно пересобрать той же командой. Проверена сборка в пустой каталог; повторная сборка дала идентичные векторы.

Подготовкой занимается `scripts/prepare.py`: `assets` скачивает файлы, `corpus` извлекает текст из PDF, `qa` собирает вопросы и эталоны, `all` выполняет все три этапа. Например, `python -m scripts.prepare qa` заново создаст открытый набор в `datasets/dev/` и авторский набор в корне задания. С параметром `--output-dir` оба набора можно записать в другую папку. Для полного восстановления удалённой папки `local/` повторите подготовку и сборку индекса из выбранного раздела выше.

## Проверки и повтор экспериментов

Проверка контрактов и оценка сохранённого итогового прогона:

```bash
python -m unittest discover -s tests -v
python -m scripts.evaluate --questions datasets/dev/dev_questions.jsonl --gold datasets/dev/dev_gold.jsonl --answers dev_answers.jsonl --trace dev_trace.jsonl --corpus local/data/corpus.jsonl --c-ref 3331.4
```

`3331.4` — средняя стоимость сохранённого локального baseline из `experiments.csv`, используемая как `C_ref`. Без `--c-ref` скрипт не начисляет баллы за эффективность (`E=0`). Для оценки нового прогона замените `--answers` и `--trace` на пути к его результатам; при сравнении с заново запущенным baseline используйте его среднюю стоимость. Автоматическая оценка проверяет варианты и ссылки, но не математическую истинность каждого обоснования.

Проверка источников без вызовов Qwen:

```bash
python -m scripts.evaluate_retrieval --index local/index --questions datasets/dev/dev_questions.jsonl --gold datasets/dev/dev_gold.jsonl --config config.json
```

Она отдельно показывает полноту источников среди кандидатов, после глобального ранжирования и в итоговом контексте. Для сохранённого прогона эти метрики находятся в `results/retrieval_metrics.json`.

Для повторения сравнения подготовьте локальные модели и индекс. В первом терминале с активированным окружением и текущей папкой `task-1/` запустите сервер:

```bash
python -m scripts.serve_local
```

Во втором терминале активируйте то же окружение, перейдите в `task-1/` и выполните:

```bash
LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=Qwen/Qwen3.5-4B SUBMISSION_ID=tfkp-comparison python -m scripts.run_experiments
```

Скрипт назначает каждому варианту свой идентификатор прогона и обновляет `experiments.csv`, `results/experiment_details.json`, `dev_answers.jsonl` и `dev_trace.jsonl`. Расход `C_ref` берётся из полученного baseline автоматически. Обычный запуск через `run_local.py` обрабатывает вопросы только итоговым вариантом и сохраняет новые результаты в `local/outputs/`.

## Как устроено решение

В индекс входят 147 страниц учебника, разбитые на 465 исходных блоков. Из них автоматически создаются 2098 поисковых окон, включая 1493 окна вокруг формул, и указатель 18 разделов. Формулы восстанавливаются по шрифтам и координатам символов в PDF. Журнал извлечения находится в `local/data/extraction_audit.jsonl`.

Сначала BM25 и E5 находят подходящие окна, а mMiniLM ранжирует до 80 исходных источников. Поиск внутри ближайшего раздела добавляет вес уже найденным кандидатам; обязательных мест для раздела нет. Два первых глобальных результата сохраняются. Точные названия теорем и структурное совпадение формулы получают приоритет. Например, интеграл со степенным знаменателем сопоставляется с общей формулой производной, содержащей факториал; конкретные ответы при этом не вычисляются. В индексе 22 записи таких признаков, извлечённые только из корпуса. Для разорванной страницей формулировки подтягивается продолжение. Произвольные соседи отключены. Выбираются шесть основных источников с бюджетом 3500 токенов Qwen.

Нормализация формул применяется только к поисковым копиям; текст источников и `block_id` сохраняются. Поисковые окна укладываются в 512 токенов E5, поэтому их конец не обрезается при кодировании. Baseline использует прежний BM25 по исходным блокам. Поиск работает на CPU, локальная генерация — через MLX на GPU.

В итоговом запросе Qwen должна проверить каждый вариант и согласовать `answer_ids` с обоснованием. Это один вызов модели. Парсер допускает одиночную точку после JSON-объекта; другие посторонние данные и второй объект не допускаются. Сырые ответы и отметка нормализации остаются в trace. Baseline сохраняет исходный промпт.

Версии моделей, параметры генерации и лимиты указаны в `config.json` и `runtime.json`. Эти настройки подготовлены для локального эксперимента.

## Ресурсы и внешние компоненты

Измерения сохранённого прогона выполнены на Apple M4 Pro с 48 ГиБ памяти, macOS 26.6.2 и Python 3.11.16. Сборка индекса заняла 45,04 с при пиковом RSS около 1,63 ГиБ; размер индекса вместе с копиями моделей поиска — 960,9 МиБ. Пик памяти MLX в журнале сервера — около 3,27 ГиБ; полный RSS сервера отдельно не измерялся. Это результаты измерений, а не минимальные системные требования. Подробности — в [отчёте](report.md) и [статистике сборки](results/build_stats.json).

Модели: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), [mMARCO mMiniLM](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1). Ревизии зафиксированы в конфигурации и runtime; для локального сервера там же указана ревизия 4-битных весов MLX.

Внешние библиотеки: [MLX-LM](https://github.com/ml-explore/mlx-lm), [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [Snowball](https://github.com/snowballstem/snowball). Код, вопросы и документация подготовлены с помощью Codex; описание этой помощи и ограничений решения находится в отчёте.
