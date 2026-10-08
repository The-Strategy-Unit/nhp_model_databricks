"""NHP Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from nhp.databricks.helpers import DatabricksData, log_execution


class DatabricksProvider(DatabricksData):
    """Load NHP data from databricks."""

    def __init__(
        self,
        spark_builder: Callable[[], SparkSession],
        data_path: str,
        year: int,
        dataset: str,
    ):
        """Initialise Databricks data loader class."""
        super().__init__(spark_builder, data_path, year, dataset)

    @property
    def spark(self) -> SparkSession:
        if self._spark is None or not self._spark.getActiveSession():
            self._spark = self._spark_builder()
        return self._spark

    @property
    def _apc(self):
        return (
            self.spark.read.parquet(f"{self._data_path}/ip")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .persist()
        )

    @log_execution
    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe.

        :return: the inpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._apc.toPandas()

    @log_execution
    def get_ip_strategies(self) -> dict[str, pd.DataFrame]:
        """Get the inpatients strategies dataframe.

        :return: the inpatients strategies dataframes
        :rtype: dict[str, pd.DataFrame]
        """
        return {
            k: self.spark.read.parquet(f"{self._data_path}/ip_{k}_strategies")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .join(self._apc, "rn", "semi")
            .toPandas()
            for k in ["activity_avoidance", "efficiencies"]
        }

    @log_execution
    def get_ip_functional_areas_beds(self) -> pd.DataFrame:
        """Get the inpatients functional areas beds dataframe.

        Returns:
            The inpatients functional areas beds dataframe.
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/ip_functional_areas_beds")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .join(self._apc, "rn", "semi")
            .toPandas()
        )

    @log_execution
    def get_ip_functional_areas_procedures(self) -> pd.DataFrame:
        """Get the inpatients functional areas procedures dataframe.

        Returns:
            The inpatients functional areas procedures dataframe.
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/ip_functional_areas_procedures")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .join(self._apc, "rn", "semi")
            .toPandas()
        )

    @log_execution
    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe.

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/op")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    @log_execution
    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe.

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/aae")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    @log_execution
    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/birth_factors")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset")
            .toPandas()
        )

    @log_execution
    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/demographic_factors")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset")
            .toPandas()
        )

    @log_execution
    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/hsa_activity_tables")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset", "fyear")
            .toPandas()
        )

    @log_execution
    def get_hsa_gams(self):
        """Get the health status adjustment gams."""
        # this is not supported in our data bricks environment currently
        raise NotImplementedError

    @log_execution
    def get_inequalities(self) -> pd.DataFrame:
        """Get the inequalities dataframe.

        Returns:
            The inequalities dataframe.
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/inequalities")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .toPandas()
        )

    def data_exists_for_model_type(self, model_type: Any) -> bool:
        """Check if data exists for a specific model type.

        Args:
            model_type: The model type to check for.

        Returns:
            True if data exists for the model type, False otherwise.
        """
        # match model_type.__name__:
        #     case "InpatientsModel":
        #         path = self._file_path("ip")
        #     case "OutpatientsModel":
        #         path = self._file_path("op")
        #     case "AaEModel":
        #         path = self._file_path("aae")
        #     case _:
        #         raise ValueError(f"Unknown model type: {model_type.__name__}")

        # return os.path.exists(path)

        return True
