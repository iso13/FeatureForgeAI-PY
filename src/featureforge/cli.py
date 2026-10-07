"""``featureforge`` command line. TS equivalent: the npm scripts in package.json.

featureforge run       → npm run cucumber[:dev|:staging]
featureforge lint      → npm run lint:gherkin
featureforge scaffold  → npm run scaffold:steps
featureforge metrics   → (new) summarize reports/run-metrics.json
featureforge packs     → (new) list installed packs
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from featureforge.core.settings import Settings, find_pyproject

DEFAULT_TAGS = "not @wip and not @performanceTest and not @demo"


def cucumber_tags_to_markers(expression: str) -> str:
    """``not @wip and (@smoke or @no-browser)`` → ``not wip and (smoke or no_browser)``."""
    return re.sub(r"@([\w-]+)", lambda m: m.group(1).replace("-", "_"), expression).strip()


def _settings() -> Settings:
    return Settings.load(find_pyproject(Path.cwd()))


def cmd_run(args: argparse.Namespace, extra: list[str]) -> int:
    import pytest

    settings = _settings()
    reports = Path(args.reports_dir) if args.reports_dir else settings.reports_dir
    for sub_dir in ("cucumber", "junit"):
        (reports / sub_dir).mkdir(parents=True, exist_ok=True)
    pytest_args = [
        "-m",
        cucumber_tags_to_markers(args.tags),
        f"--cucumberjson={reports / 'cucumber' / 'cucumber_report.json'}",
        f"--junitxml={reports / 'junit' / 'junit.xml'}",
    ]
    if args.env:
        pytest_args.append(f"--ff-env={args.env}")
    if args.headed:
        pytest_args.append("--ff-headed")
    if args.reports_dir:
        pytest_args.append(f"--ff-reports-dir={args.reports_dir}")
    for pack in args.pack or []:
        pytest_args.append(f"--ff-pack={pack}")
    if args.parallel and args.parallel > 1:
        pytest_args += ["-n", str(args.parallel)]  # requires pytest-xdist
    targets = [arg for arg in extra if Path(arg.split("::")[0]).exists()]
    if not targets and Path("tests/bdd").is_dir():
        pytest_args.append("tests/bdd")  # BDD scenarios only; plain `pytest` runs everything
    return int(pytest.main([*pytest_args, *extra]))


def cmd_lint(args: argparse.Namespace) -> int:
    from featureforge.tools.lint import lint_paths

    paths = [Path(p) for p in args.paths] or [_default_features_dir()]
    issues = lint_paths(paths, _settings().lint)
    root = Path.cwd()
    for issue in issues:
        try:
            print(issue.render(root))
        except ValueError:
            print(issue.render())
    print(f"{len(issues)} issue(s)" if issues else "All feature files passed")
    return 1 if issues else 0


def cmd_scaffold(args: argparse.Namespace) -> int:
    from featureforge.tools.scaffold import scaffold

    steps = [Path(p) for p in args.steps] or [Path("src"), Path("tests")]
    missing, source = scaffold([Path(p) for p in args.features], steps)
    if not missing:
        print("Every step already has a definition.")
        return 0
    if args.out:
        out = Path(args.out)
        if out.exists() and not args.force:
            print(f"{out} exists; pass --force to overwrite", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(source, encoding="utf-8")
        print(f"Wrote {len(missing)} step stub(s) to {out}")
    else:
        print(source)
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    from featureforge.core.metrics import RunMetrics

    path = Path(args.path) if args.path else _settings().reports_dir / "run-metrics.json"
    if not path.is_file():
        print(f"No metrics at {path}; run `featureforge run` first.", file=sys.stderr)
        return 2
    metrics = RunMetrics.read(path)
    print(metrics.render())
    for result in metrics.slowest(args.top):
        print(f"  {result.duration_s:8.3f}s  {result.outcome:<7} {result.label}")
    return 0


def cmd_packs(_: argparse.Namespace) -> int:
    from featureforge.packs import discover_packs

    for pack in discover_packs().values():
        print(f"{pack.name:<12} {pack.description}")
    return 0


def _default_features_dir() -> Path:
    from featureforge.packs import discover_packs

    packs = discover_packs()
    return packs["ai"].features_dir if "ai" in packs else Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="featureforge", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run BDD scenarios (extra args after -- go to pytest)")
    run.add_argument("--tags", default=DEFAULT_TAGS, help="Cucumber-style tag expression")
    run.add_argument("--env", help="Environment key from [tool.featureforge.environments]")
    run.add_argument("--headed", action="store_true")
    run.add_argument("--pack", action="append")
    run.add_argument("--reports-dir")
    run.add_argument("--parallel", type=int, default=1)

    lint = sub.add_parser("lint", help="Lint feature files")
    lint.add_argument("paths", nargs="*")

    scaffold = sub.add_parser("scaffold", help="Generate stubs for undefined steps")
    scaffold.add_argument("features", nargs="+")
    scaffold.add_argument("--steps", action="append", default=[], help="Step code to scan")
    scaffold.add_argument("--out", help="Write stubs to this file instead of stdout")
    scaffold.add_argument("--force", action="store_true")

    metrics = sub.add_parser("metrics", help="Summarize the last run")
    metrics.add_argument("path", nargs="?")
    metrics.add_argument("--top", type=int, default=5)

    sub.add_parser("packs", help="List installed packs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in raw:
        split = raw.index("--")
        raw, extra = raw[:split], raw[split + 1 :]
    args = build_parser().parse_args(raw)
    if args.command == "run":
        return cmd_run(args, extra)
    handlers = {
        "lint": cmd_lint,
        "scaffold": cmd_scaffold,
        "metrics": cmd_metrics,
        "packs": cmd_packs,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
