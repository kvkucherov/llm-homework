"""One-command local server, official-engine demo, or course reference evaluation."""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
REFERENCE_NAMES = ("novice", "apprentice", "journeyman", "expert")


def read_references(path):
    if not path.is_file():
        raise ValueError("Нет references.json. Скопируйте references.example.json и укажите API-адреса эталонов. "
                         "Ссылки на профили арены не являются API-адресами.")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) != set(REFERENCE_NAMES):
        raise ValueError("Нужны все четыре ключа: " + ", ".join(REFERENCE_NAMES))
    result = {}
    for name in REFERENCE_NAMES:
        url = data[name]
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"Не указан API-адрес эталона {name}. Эталоны не входят в репозиторий.")
        url = url.rstrip("/")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError(f"Неверный API-адрес {name}")
        if parsed.hostname in ("arena.battlesnake.com", "play.battlesnake.com"):
            raise ValueError(f"{name}: нужна база API бота, а не страница сайта Battlesnake")
        result[name] = url
    if len(set(result.values())) != 4:
        raise ValueError("Адреса четырёх эталонов должны отличаться")
    return result


def check_endpoint(url):
    with urllib.request.urlopen(url + "/", timeout=5) as response:
        metadata = json.load(response)
    if not isinstance(metadata, dict) or metadata.get("apiversion") != "1":
        raise ValueError(f"{url} не отвечает как Battlesnake API v1")


@contextlib.contextmanager
def local_server(engine, budget=280, policy=None, depth=32, weights=None):
    from http.server import ThreadingHTTPServer
    from snake.server import handler_factory
    options = {"budget_ms":budget,"depth":depth,"policy":policy}
    if weights is not None:
        options["weights"] = weights
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(engine, **options))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        check_endpoint(url)
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def course_points(summary):
    if sum(summary.values()) != 11:
        raise ValueError("Для зачётной серии нужно ровно 11 завершённых дуэлей")
    return 15 if summary["win"] > summary["loss"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="собрать и запустить бота для проверки преподавателем")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--budget-ms", type=int, default=280)
    demo = sub.add_parser("demo", help="одна команда: сборка, CLI, серверы и тренировочные дуэли")
    demo.add_argument("--games", type=int, default=11)
    demo.add_argument("--budget-ms", type=int, default=20,
                      help="время на ход; 20 мс для быстрой тренировки, 280 мс для рабочего режима")
    demo.add_argument("--seed", type=int, default=5000)
    demo.add_argument("--policy", choices=("greedy", "cautious", "territory", "hunter"), default="territory")
    course = sub.add_parser("course", help="11 дуэлей против каждого из четырёх настоящих эталонов")
    course.add_argument("--references", type=Path, default=Path("references.json"))
    course.add_argument("--seed", type=int, default=7000)
    args = parser.parse_args()
    # Resolve caller-supplied paths before entering the project, so invocation
    # from the repository root behaves exactly like invocation from task-2.
    reference_path = args.references.resolve() if args.command == "course" else None
    if reference_path and not args.references.is_absolute() and str(args.references) == "references.json":
        reference_path = ROOT / "references.json"
    os.chdir(ROOT)
    endpoints = read_references(reference_path) if reference_path else None
    if args.command == "demo" and args.games < 1:
        raise ValueError("games must be positive")
    subprocess.run(["make", "all"], check=True)
    if args.command == "serve":
        return subprocess.call([sys.executable, "-m", "snake.server", "--host", args.host,
                                "--port", str(args.port), "--budget-ms", str(args.budget_ms)])
    cli = ROOT / "vendor/battlesnake/battlesnake"
    if not cli.is_file():
        from tools.install_cli import main as install_cli
        install_cli()
    from snake.engine import Engine
    from tools.official import run_series
    engine = Engine()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = ROOT / "local" / stamp
    with local_server(engine, getattr(args, "budget_ms", 280)) as ours:
        if args.command == "demo":
            print("ТРЕНИРОВКА: соперник написан в этом проекте. Это не эталон курса и не зачёт на 60 баллов.", flush=True)
            with local_server(engine, policy=args.policy) as opponent:
                report = run_series(cli, ours, opponent, args.games, args.seed, directory=directory)
            print("Итог тренировки:", report["summary"])
        else:
            for name, endpoint in endpoints.items():
                check_endpoint(endpoint)
            total = 0
            report = {"notice": "Идентичность эталонов определяется предоставленными API-адресами; "
                                  "это локальная проверка, не выставленная преподавателем оценка.", "references": {}}
            for index, (name, endpoint) in enumerate(endpoints.items()):
                print(f"\n{name}: 11 дуэлей", flush=True)
                result = run_series(cli, ours, endpoint, 11, args.seed + index * 100, directory=directory / name)
                points = course_points(result["summary"])
                report["references"][name] = {"url": endpoint, "summary": result["summary"], "points": points}
                total += points
            report["points_if_endpoints_are_authentic_references"] = total
            (directory / "course-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"Условный результат для указанных эталонов: {total}/60")
        print("Отчёт и повторы:", directory)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(2)
