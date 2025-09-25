"""NHP Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from typing import Any, Callable

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from nhp.model.data import Data


class DatabricksICB(Data):
    """Load NHP data from databricks."""

    def __init__(self, spark: SparkSession, data_path: str, year: int, icb: str):
        """Initialise Databricks data loader class."""
        self._spark = spark
        self._data_path = data_path
        self._year = year
        self._icb = icb

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
        return lambda fyear, icb: DatabricksICB(spark, data_path, fyear, icb)

    @property
    def _apc(self):
        return (
            self._spark.read.parquet(f"{self._data_path}/ip")
            .filter(F.col("icb") == self._icb)
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
            .filter(F.col("icb") == self._icb)
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
            .filter(F.col("icb") == self._icb)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """
        # which year of the ONS population projections to use
        projection_year = 2022
        # load the tables
        births_df = self._spark.read.table("nhp.population_projections.births").filter(
            F.col("projection_year") == projection_year
        )
        catchments_df = self._spark.read.table("nhp.reference.icb_catchments").filter(
            F.col("icb") == self._icb
        )
        # join and aggregate
        return (
            births_df.join(catchments_df, "area_code")
            .groupBy("age", "sex", "projection")
            .agg(F.sum(F.col("value") * F.col("pcnt")).alias("value"))
            .toPandas()
        )

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        # which year of the ONS population projections to use
        projection_year = 2022
        # load the tables
        demographics_df = self._spark.read.table("nhp.population_projections.demographics").filter(
            F.col("projection_year") == projection_year
        )
        catchments_df = self._spark.read.table("nhp.reference.icb_catchments").filter(
            F.col("icb") == self._icb
        )
        # join and aggregate
        return (
            demographics_df.join(catchments_df, "area_code")
            .groupBy("age", "sex", "projection")
            .agg(F.sum(F.col("value") * F.col("pcnt")).alias("value"))
            .toPandas()
        )

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.table("nhp.default.hsa_activity_tables_icb")
            .filter(F.col("icb") == self._icb)
            .filter(F.col("fyear") == self._year * 100 + (self._year + 1) % 100)
            .groupBy("hsagrp", "sex", "age")
            .agg(F.mean("activity").alias("activity"))
            .toPandas()
        )

    def get_hsa_gams(self):
        """Get the health status adjustment gams."""
        # this is not supported in our data bricks environment currently
        raise NotImplementedError
