# nhp.databricks

Databricks runner and terminal UI for executing `nhp-model` scenarios against
Databricks-backed datasets.

You will need [`uv`](https://docs.astral.sh/uv/) installed.

## What this repo provides

- A reusable run wrapper (`Run`) that executes model runs and stores metadata.
- A terminal UI entry point (`nhp_databricks_tui`) for interactive runs.
- Dataset-specific data adapters for `provider`, `icb`, and `national` modes.
- A notebook example at `notebooks/run.ipynb`.

## Requirements

- Python 3.12+
- `uv` (recommended for environment and dependency management)
- Access to a Databricks cluster
- Access to model data storage

## Set up Databricks authentication

Open up a browser tab and log into the Databricks website.
Copy the databricks domain name to your clipboard (remove anything in the URL
after the domain suffix (.com/.net)).

Run

```powershell
databricks auth login
```

Select option 2 ("enter a host URL manually"), and then paste the domain name.

## Optional: pin to a specific Python version

`uv` will install the version if not already available.

For example:

```powershell
uv python pin 3.14
```

This will create (or update) a `.python-version` file in the repository root.

## Install

From the repository root:

```powershell
uv sync
```

This creates `.venv` and installs project dependencies from `pyproject.toml`.

## Configuration

Create a `.env` file in the repository root (or export env vars in your shell):

```env
DATA_PATH=<base data path used by the model>
DATABRICKS_CLUSTER_ID=<cluster id>
```

A `.env.sample` file is provided in the repository for your convenience.

### Notes

- `DATA_PATH` is required by the TUI and is combined with a model data version
	such as `v6.0.0`.
- `DATABRICKS_CLUSTER_ID` is required when creating the Databricks Spark session.

If you choose to upload results from the TUI, additional Databricks workspace
secrets are required by `Run.upload_results()`.

## Run the TUI

Use the console script directly:

```powershell
uv run nhp_databricks_tui
```

You can also run the module directly:

```powershell
uv run -m nhp.databricks.tui
```

It may take a few seconds for the TUI to appear at the terminal.

If these run commands do not work, you can try running

```powershell
uv pip install -e .
```

as an alternative to `uv sync`, which may unblock things.
But the pip approach is not optimal.

## TUI flow

When launched, the TUI will:

1. Ask for run mode (`provider`, `icb`, or `national`).
2. Ask whether to load params from a file under `param_files/`.
3. Ask for model data version (default: `v6.0.0`).
4. Run the model and print a summary table and metadata.
5. Optionally upload results.

## Params files

- Put params files in `param_files/` for quick selection in the TUI.
- If none are present, you can manually enter a path.

## Notebook usage

The repository includes `notebooks/run.ipynb` for an interactive run flow.

## Troubleshooting

### `can't open file '...\\nhp_databricks_tui'`

Use `uv run nhp_databricks_tui` instead of `uv run python nhp_databricks_tui`.

### `DATA_PATH environment variable is not set`

Set `DATA_PATH` in `.env` or your shell.

### `DATABRICKS_CLUSTER_ID environment variable is not set`

Set `DATABRICKS_CLUSTER_ID` in `.env` or your shell.

### Entry point not found

Recreate the environment and install dependencies:

```powershell
uv sync
```
