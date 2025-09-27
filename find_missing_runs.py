import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple


SCRIPT_PATTERN = re.compile(r"^run_(?P<prompt>[^_]+)_(?P<attack>[^_]+)_(?P<mitigation>[^.]+)\.sh$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find experiments without successful result files and list scripts to re-run.")
    parser.add_argument("--scripts_dir", default="run_scripts", help="Directory containing generated .sh run scripts")
    parser.add_argument("--results_dir", default="data/qwen3_0.6b_data", help="Directory containing results_*.json files")
    return parser.parse_args()


def expected_result_filename(prompt: str, attack: str, mitigation: str) -> str:
    return f"results_{prompt}_{attack}_{mitigation}.json"


def is_result_successful(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    # Accept either the new wrapped structure or legacy list
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list) and len(results) > 0:
            return True
        # If it's dict but no results list, treat as failed
        return False
    if isinstance(data, list):
        return len(data) > 0
    return False


def collect_scripts(scripts_dir: Path) -> List[Tuple[Path, str, str, str]]:
    out: List[Tuple[Path, str, str, str]] = []
    for path in sorted(scripts_dir.glob("run_*.sh")):
        m = SCRIPT_PATTERN.match(path.name)
        if not m:
            continue
        prompt = m.group("prompt")
        attack = m.group("attack")
        mitigation = m.group("mitigation")
        out.append((path, prompt, attack, mitigation))
    return out


def main() -> None:
    args = parse_args()
    scripts_dir = Path(args.scripts_dir)
    results_dir = Path(args.results_dir)

    if not scripts_dir.exists():
        raise FileNotFoundError(f"Scripts directory not found: {scripts_dir}")
    if not results_dir.exists():
        print(f"Warning: Results directory does not exist yet: {results_dir}")

    scripts = collect_scripts(scripts_dir)
    missing: List[Tuple[Path, Path]] = []

    for script_path, prompt, attack, mitigation in scripts:
        expected_name = expected_result_filename(prompt, attack, mitigation)
        result_path = results_dir / expected_name
        if not is_result_successful(result_path):
            missing.append((script_path, result_path))

    if not missing:
        print("All experiments have successful results.")
        return

    print("The following experiments are missing or failed. Re-run these scripts:")
    for script_path, result_path in missing:
        print(f"- {script_path}    (expected result: {result_path.name})")

    # Also write to a helper file for convenience
    rerun_list = results_dir / "rerun_scripts.txt"
    rerun_list.parent.mkdir(parents=True, exist_ok=True)
    with open(rerun_list, "w", encoding="utf-8", newline="\n") as f:
        for script_path, _ in missing:
            f.write(str(script_path).replace("\\", "/") + "\n")
    print(f"\nSaved list to: {rerun_list}")


if __name__ == "__main__":
    main()


