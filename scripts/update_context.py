"""Bundles custom project files (scripts/, src/, pyproject.toml) and experiment logs into CONTEXT.md."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
INCLUDE_DIRS = ["scripts", "src"]
INCLUDE_FILES = ["pyproject.toml"]
EXP_DIR = ROOT / "experiments"
OUTPUT_FILE = ROOT / "CONTEXT.md"


def main():
    lines = ["# Balatro RL Project — Live Codebase Snapshot\n"]

    # 1. Record directory tree (excluding jackdaw-balatro engine & .venv)
    lines.append("## Project Structure\n```text")
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if any(
            part.startswith(".")
            or part in ("jackdaw-balatro", "__pycache__", "logs", "models")
            or p.name.startswith("events.out.tfevents")
            for part in rel.parts
        ):
            continue
        if p.is_file():
            lines.append(str(rel))
    lines.append("```\n")

    # 2. Dump root config files
    for fname in INCLUDE_FILES:
        fpath = ROOT / fname
        if fpath.exists():
            lines.append(f"## `{fname}`\n```toml\n{fpath.read_text()}\n```\n")

    # 3. Dump all custom Python scripts in scripts/ and src/
    for dname in INCLUDE_DIRS:
        dpath = ROOT / dname
        if not dpath.exists():
            continue
        for py_file in sorted(dpath.rglob("*.py")):
            rel = py_file.relative_to(ROOT)
            lines.append(f"## `{rel}`\n```python\n{py_file.read_text()}\n```\n")

    # 4. Dump latest experiment CSV logs (header + first 5 + last 20 rows)
    if EXP_DIR.exists():
        for csv_file in sorted(EXP_DIR.rglob("progress.csv")):
            rel = csv_file.relative_to(ROOT)
            csv_lines = [line for line in csv_file.read_text().splitlines() if line.strip()]
            if not csv_lines:
                continue
            if len(csv_lines) <= 30:
                snippet = "\n".join(csv_lines)
            else:
                header = csv_lines[0]
                first_rows = "\n".join(csv_lines[1:6])
                last_rows = "\n".join(csv_lines[-20:])
                snippet = f"{header}\n{first_rows}\n... ({len(csv_lines) - 26} intermediate rows omitted; see raw {rel}) ...\n{last_rows}"
            lines.append(f"## `{rel}` (Training Progress Summary)\n```csv\n{snippet}\n```\n")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated {OUTPUT_FILE.name} ({len(lines)} sections)")


if __name__ == "__main__":
    main()