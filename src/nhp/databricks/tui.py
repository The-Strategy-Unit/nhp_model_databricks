import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from nhp.model.params import load_params, load_sample_params
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from nhp.databricks.icb import DatabricksICB
from nhp.databricks.national import DatabricksNational
from nhp.databricks.provider import DatabricksProvider
from nhp.databricks.run import Run

DEFAULT_MODEL_DATA_VERSION = "v6.0.0"


def _ask_required_value(prompt_text: str) -> str:
    while True:
        value = Prompt.ask(prompt_text).strip()
        if value:
            return value


def _choose_mode(console: Console) -> str:
    mode = Prompt.ask(
        "Select run mode",
        choices=["provider", "icb", "national"],
        default="national",
    )
    return mode


def _choose_params_file(console: Console, folder: Path) -> Path:
    files = (
        sorted([p for p in folder.glob("*") if p.is_file()]) if folder.exists() else []
    )

    if not files:
        manual = Prompt.ask(
            "No files found in param_files. Enter a params file path"
        ).strip()
        file_path = Path(manual).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Params file not found: {file_path}")
        return file_path

    console.print("\nAvailable params files:")
    for idx, file_path in enumerate(files, start=1):
        console.print(f"  {idx}. {file_path.name}")
    console.print("  m. Enter a path manually")

    choice = Prompt.ask(
        "Choose params file",
        choices=[str(i) for i in range(1, len(files) + 1)] + ["m"],
        default="1",
    )

    if choice == "m":
        manual = Prompt.ask("Enter a params file path").strip()
        file_path = Path(manual).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
    else:
        file_path = files[int(choice) - 1]

    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Params file not found: {file_path}")
    return file_path


def _load_run_params(
    console: Console, mode: str, workspace_root: Path
) -> dict[str, Any]:
    use_params_file = Confirm.ask("Use a params file?", default=False)

    if use_params_file:
        params_file = _choose_params_file(console, workspace_root / "param_files")
        console.print(f"Loading params from {params_file}")
        return load_params(str(params_file))

    if mode == "national":
        dataset = "national"
        console.print("Using dataset='national' for national mode")
    else:
        dataset = _ask_required_value("Enter dataset")

    scenario = _ask_required_value("Enter scenario name")

    return load_sample_params(dataset=dataset, scenario=scenario)


def _choose_data_class(mode: str):
    if mode == "provider":
        return DatabricksProvider
    if mode == "icb":
        return DatabricksICB
    return DatabricksNational


def _build_summary(
    mode: str, default_df: pd.DataFrame, sample_rate: float | None
) -> pd.DataFrame:
    required = {"pod", "measure", "model_run", "value"}
    if not required.issubset(default_df.columns):
        return default_df.head(20)

    summary_df = default_df.copy()
    if mode == "national" and sample_rate and sample_rate != 0:
        summary_df = summary_df.assign(value=lambda x: x["value"] / sample_rate)

    summary_df = (
        summary_df.assign(
            run_type=lambda x: (
                x["model_run"].eq(0).map({True: "baseline", False: "principal"})
            )
        )
        .groupby(["pod", "measure", "run_type"], as_index=False)["value"]
        .mean()
        .pivot(index=["pod", "measure"], columns="run_type", values="value")
        .reset_index()
    )

    if "principal" not in summary_df.columns:
        summary_df["principal"] = pd.NA
    if "baseline" not in summary_df.columns:
        summary_df["baseline"] = pd.NA

    return summary_df.assign(diff=lambda x: x["principal"] - x["baseline"])


def _render_dataframe(console: Console, df: pd.DataFrame, title: str) -> None:
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include="number").columns
    numeric_col_set = set(numeric_cols)
    if len(numeric_cols) > 0:
        display_df[numeric_cols] = display_df[numeric_cols].round(0)

    table = Table(title=title)
    for col in display_df.columns:
        justify = "right" if col in numeric_col_set else "left"
        table.add_column(str(col), justify=justify)

    for _, row in display_df.fillna("").iterrows():
        table.add_row(*[str(v) for v in row.tolist()])

    console.print(table)


def run_tui() -> None:
    load_dotenv()
    console = Console()
    workspace_root = Path.cwd()

    mode = _choose_mode(console)
    params = _load_run_params(console, mode, workspace_root)

    model_data_version = Prompt.ask(
        "Model data version",
        default=DEFAULT_MODEL_DATA_VERSION,
    ).strip()

    data_root = os.getenv("DATA_PATH")
    if not data_root:
        raise ValueError("DATA_PATH environment variable is not set.")

    data_path = f"{data_root}/{model_data_version}"

    data_class = _choose_data_class(mode)
    run_kwargs: dict[str, Any] = {}
    if mode == "national":
        maybe_sample_rate = params.get("sample_rate")
        if isinstance(maybe_sample_rate, (int, float)):
            run_kwargs["sample_rate"] = float(maybe_sample_rate)

    console.print("\nRunning model...")
    run = Run(params=params, data_class=data_class, data_path=data_path, **run_kwargs)
    run.go()

    default_results = run.results.get("default")
    if isinstance(default_results, pd.DataFrame):
        sample_rate = run_kwargs.get("sample_rate") if mode == "national" else None
        summary = _build_summary(mode, default_results, sample_rate)
        dataset_value = params.get("dataset", "unknown")
        _render_dataframe(
            console,
            summary,
            title=f"Run Summary ({mode}, dataset={dataset_value})",
        )
    else:
        console.print("No tabular default results available to summarize.")

    meta_table = Table(title="Run Metadata")
    meta_table.add_column("Field")
    meta_table.add_column("Value")
    for key in [
        "model_run_id",
        "model_run_start_time",
        "model_run_end_time",
        "model_run_elapsed_time_seconds",
        "status",
    ]:
        meta_table.add_row(key, str(run.run_metadata.get(key, "")))
    console.print(meta_table)

    if Confirm.ask("Upload results now?", default=False):
        console.print("Uploading results...")
        run.upload_results()
        console.print("Upload complete.")
    else:
        console.print("Upload skipped.")


def main() -> None:
    try:
        run_tui()
    except Exception as exc:
        Console().print(f"[red]Error:[/red] {exc}")
        raise


if __name__ == "__main__":
    main()
