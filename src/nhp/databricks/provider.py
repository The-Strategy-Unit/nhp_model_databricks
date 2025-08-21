"""NHP Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from typing import Any, Callable

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from nhp.model.data import Data


class DatabricksProvider(Data):
    """Load NHP data from databricks."""

    def __init__(self, spark: SparkSession, data_path: str, year: int, dataset: str):
        """Initialise Databricks data loader class."""
        self._spark = spark
        self._data_path = data_path
        self._year = year
        self._dataset = dataset

    @staticmethod
    def create(spark: SparkSession, data_path: str) -> Callable[[int, str], Any]:
        """Create Databricks object.

        :param spark: a SparkSession for selecting data
        :type spark: SparkSession
        :param data_path: the path to where the parquet files are stored
        :type data_path: str
        :return: a function to initialise the object
        :rtype: Callable[[str, str], Databricks]
        """
        return lambda fyear, dataset: DatabricksProvider(
            spark, data_path, fyear, dataset
        )

    @property
    def _apc(self):
        return (
            self._spark.read.parquet(f"{self._data_path}/ip")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .persist()
        )

    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe.

        :return: the inpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._apc.toPandas()

    def get_ip_strategies(self) -> dict[str, pd.DataFrame]:
        """Get the inpatients strategies dataframe.

        :return: the inpatients strategies dataframes
        :rtype: dict[str, pd.DataFrame]
        """
        return {
            k: self._spark.read.parquet(f"{self._data_path}/ip_{k}_strategies")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .join(self._apc, "rn", "semi")
            .toPandas()
            for k in ["activity_avoidance", "efficiencies"]
        }

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe.

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/op")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe.

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/aae")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/birth_factors")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset")
            .toPandas()
        )

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/demographic_factors")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset")
            .toPandas()
        )

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/hsa_activity_tables")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset", "fyear")
            .toPandas()
        )

    def get_hsa_gams(self):
        """Get the health status adjustment gams."""
        # this is not supported in our data bricks environment currently
        raise NotImplementedError
