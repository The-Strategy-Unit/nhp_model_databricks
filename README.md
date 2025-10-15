# The New Hospital Programme Demand Model - Databricks implementation

This repository contains code for the Databricks implementation of the NHP model.

The notebooks in this repository can be used to run provider, ICB, and national level runs of the NHP model.

## Explanation of the NHP model Databricks implementation

nhp_model code is a package on GitHub. We tag specific versions of the nhp_model, e.g. v4.2.1.

This package has as its package dependency the tagged version of nhp_model code, available from GitHub.
So for example, you can specify v4.2.1 or v4.1.0 of nhp_model. This is currently specified in the `pyproject.toml` file.

This repository contains specific Data loading classes for ICB, national and provider level model runs.
These tell the nhp_model code to load the data for these runs from Databricks.

This repository also has workflow yaml file in the `resources` folder that tell Databricks what to do.
The workflow is:

1. Look on Azure for params files in a specific folder, then
1. Run a specific notebook using those params.

nhp_model_databricks uses the same extracted parquet data as the provider level model, just samples from it differently.
It reads some of the reference tables directly from Databricks.

## How to use the NHP model Databricks implementation

This implementation is still a work in progress.
We're keeping notes for internal use only in the [nhp_products Wiki](https://github.com/The-Strategy-Unit/nhp_products/wiki/How-to-run-the-ICB-and-national-level-models).
