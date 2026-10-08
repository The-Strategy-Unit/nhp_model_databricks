"""NHP National Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from collections.abc import Callable
from random import randint
from typing import Any

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from nhp.databricks.helpers import DatabricksData, log_execution


class DatabricksNational(DatabricksData):
    """Load NHP data from databricks."""

    def __init__(
        self,
        spark_builder: Callable[[], SparkSession],
        data_path: str,
        year: int,
        dataset: str = "national",
        sample_rate: float = 0.001,
        seed: int | None = None,
    ):
        """Initialise DatabricksNational data loader class."""
        super().__init__(spark_builder, data_path, year, dataset)

        self._sample_rate = sample_rate
        self._seed = seed or randint(0, 2**12)

        self._apc = (
            self.spark.read.parquet(f"{self._data_path}/ip")
            .filter(F.col("fyear") == self._year)
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .sample(fraction=self._sample_rate, seed=self._seed)
            .persist()
        )

    @property
    def spark(self) -> SparkSession:
        if self._spark is None or not self._spark.getActiveSession():
            self._spark = self._spark_builder()
        return self._spark

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
        op = self.spark.read.parquet(f"{self._data_path}/op")

        return (
            self.spark.read.parquet(f"{self._data_path}/op")
            .filter(F.col("fyear") == self._year)
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .withColumn("icb", F.lit("NATIONAL"))
            # TODO: temporary fix, see #353
            .withColumn("sushrg_trimmed", F.lit("HRG"))
            .withColumn("imd_quintile", F.lit(0))
            .groupBy(
                op.drop("index", "fyear", "attendances", "tele_attendances").columns
            )
            .agg(
                (F.sum("attendances") * self._sample_rate).alias("attendances"),
                (F.sum("tele_attendances") * self._sample_rate).alias(
                    "tele_attendances"
                ),
            )
            # TODO: how do we make this stable? at the moment we can't use full model results with
            # national
            .withColumn("rn", F.expr("uuid()"))
            .toPandas()
        )

    @log_execution
    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe.

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        aae = self.spark.read.parquet(f"{self._data_path}/aae")

        return (
            self.spark.read.parquet(f"{self._data_path}/aae")
            .filter(F.col("fyear") == self._year)
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .withColumn("icb", F.lit("NATIONAL"))
            .groupBy(aae.drop("index", "fyear", "arrivals").columns)
            .agg((F.sum("arrivals") * self._sample_rate).alias("arrivals"))
            # TODO: how do we make this stable? at the moment we can't use full model results with
            # national
            .withColumn("rn", F.expr("uuid()"))
            .toPandas()
        )

    @log_execution
    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        :param projection_year: Which projection year to use?
        :type projection_year: int, defaults to 2022
        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """
        births_df = (
            self.spark.read.parquet(f"{self._data_path}/birth_factors/")
            .filter(F.col("fyear") == self._year)
            .filter(~F.col("variant").startswith("custom"))
        )

        years = [i for i in births_df.columns if i.startswith("20")]
        years_str = ", ".join([f"'{i}', `{i}`" for i in years])
        expr = f"stack({len(years)}, {years_str}) as (year, value)"

        return (
            births_df.selectExpr("variant", "age", "sex", expr)
            .groupBy("variant", "age", "sex")
            .pivot("year")
            .agg(F.sum("value"))
            .orderBy("variant", "age", "sex")
            .withColumnRenamed("projection", "variant")
            .toPandas()
        )

    @log_execution
    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :param projection_year: Which projection year to use?
        :type projection_year: int, defaults to 2022
        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        demog_df = (
            self.spark.read.parquet(f"{self._data_path}/demographic_factors/")
            .filter(F.col("fyear") == self._year)
            .filter(~F.col("variant").startswith("custom"))
        )

        years = [i for i in demog_df.columns if i.startswith("20")]
        years_str = ", ".join([f"'{i}', `{i}`" for i in years])
        expr = f"stack({len(years)}, {years_str}) as (year, value)"

        return (
            demog_df.selectExpr("variant", "age", "sex", expr)
            .groupBy("variant", "age", "sex")
            .pivot("year")
            .agg(F.sum("value"))
            .orderBy("variant", "age", "sex")
            .withColumnRenamed("projection", "variant")
            .toPandas()
        )

    @log_execution
    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self.spark.read.table(
                "udal_lake_mart.newhospitalprogramme.default_hsa_activity_tables_national"
            )
            .filter(F.col("fyear") == self._year * 100 + (self._year + 1) % 100)
            .groupBy("hsagrp", "sex", "age")
            .agg(F.mean("activity").alias("activity"))
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
        return True
