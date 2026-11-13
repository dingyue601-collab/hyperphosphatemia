"""Run numerical analyses from institutionally prepared, deidentified inputs."""
import argparse
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STAGES = ("prepare", "benchmark", "reduce", "final", "evaluate", "explain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Directory containing the four cleaned RDS inputs.")
    parser.add_argument("--workspace", type=Path, required=True, help="Private working directory outside this release.")
    parser.add_argument("--stage", choices=("inspect", "all") + STAGES, default="inspect")
    parser.add_argument("--rscript", default=shutil.which("Rscript"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace == HERE or HERE in workspace.parents:
        raise ValueError("Keep data and generated results outside the public release directory.")
    for path in HERE.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    if args.stage == "inspect":
        print("Python syntax passed. Analysis order:", " -> ".join(STAGES))
        print("Inspect does not read patient data, fit models or install dependencies.")
        return
    if args.data is None or args.rscript is None:
        parser.error("--data and an available --rscript are required for analyses.")
    required = ("Development_inputs.rds", "Testing_target_frame.rds", "Membership.rds", "Validation_labels.rds")
    if any(not (args.data / name).is_file() for name in required):
        raise FileNotFoundError("The cleaned input directory must contain: " + ", ".join(required))
    cleaned = workspace / "data/cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    for name in required:
        target = cleaned / name
        if (args.data / name).resolve() != target.resolve():
            shutil.copy2(args.data / name, target)
    shutil.copytree(HERE / "analysis", workspace / "analysis", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(HERE / "calculator/vendor", workspace / "calculator/vendor", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    analysis = workspace / "results/analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(HYPERPHOS_RUN=str(workspace), HYPERPHOS_PYTHON=args.python,
               HYPERPHOS_R=args.rscript, HYPERPHOS_OUTPUT=str(analysis),
               PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")

    def r(name, *arguments):
        subprocess.run([args.rscript, str(workspace / "analysis" / name), *arguments], env=env, check=True)

    def python(name, *arguments, output=None):
        selected_env = dict(env)
        if output:
            selected_env["HYPERPHOS_OUTPUT"] = str(output)
        subprocess.run([args.python, "-X", "utf8", str(workspace / "analysis" / name), *arguments], env=selected_env, check=True)

    selected = STAGES if args.stage == "all" else (args.stage,)
    for stage in selected:
        print("Stage:", stage, flush=True)
        if stage == "prepare":
            r("prepare_predictors.R")
            r("weight_select.R", "run")
            r("models.R", "export")
            r("test_weights.R")
        elif stage == "benchmark":
            for family in ("Logistic", "RF", "SVM", "XGBoost", "LightGBM"):
                r("models.R", family)
            if not (analysis / "CatBoost/Device_benchmark.json").is_file():
                r("catboost.R", "benchmark")
            r("catboost.R")
            python("tune_tabm.py")
            r("compare.R")
        elif stage == "reduce":
            python("subset_search.py")
        elif stage == "final":
            r("fixed_two_inputs.R")
            r("final_comparators.R")
            python("final_tabm.py", "full")
            python("final_tabm.py", "two")
            r("nonipw_inputs.R")
            python("tune_tabm.py", output=analysis / "nonIPW_sensitivity")
            python("final_tabm.py", "nonipw")
            r("develop_two.R")
        elif stage == "evaluate":
            r("evaluate.R")
        elif stage == "explain":
            python("explain.py")
    print("Numerical results:", analysis)


if __name__ == "__main__":
    main()
