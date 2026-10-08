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


class DatabricksICB(DatabricksData):
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
            .filter(F.col("icb") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumn("sitetret", F.col("dataset"))
            .withColumn("dataset", F.lit(self._dataset))
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
            .filter(F.col("fyear") == self._year)
            .join(self._apc, "rn", "semi")
            .toPandas()
            for k in ["activity_avoidance", "efficiencies"]
        }

    @log_execution
    def get_ip_functional_areas_beds(self) -> pd.DataFrame:
        """Get the inpatients functional areas beds dataframe.

        :return: the inpatients functional areas beds dataframe
        :rtype: pd.DataFrame
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/ip_functional_areas_beds")
            .filter(F.col("fyear") == self._year)
            .groupBy("rn", "functional_area")
            .agg(
                F.sum("group_los").alias("group_los"),
                F.sum("episodes").alias("episodes"),
                F.sum("los_total").alias("los_total"),
                F.sum("group_pcnt").alias("group_pcnt"),
            )
            .withColumn("sitetret", F.lit("-"))
            .toPandas()
        )

    @log_execution
    def get_ip_functional_areas_procedures(self) -> pd.DataFrame:
        """Get the inpatients functional areas procedures dataframe.

        :return: the inpatients functional areas procedures dataframe
        :rtype: pd.DataFrame
        """
        return (
            self.spark.read.parquet(f"{self._data_path}/ip_functional_areas_procedures")
            .filter(F.col("fyear") == self._year)
            .groupBy("rn", "functional_area")
            .agg(F.sum("count").alias("count"))
            .withColumn("sitetret", F.lit("-"))
            .toPandas()
        )

    @log_execution
    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe.

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        op_df = (
            self.spark.read.parquet(f"{self._data_path}/op")
            .filter(F.col("icb") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumn("sitetret", F.col("dataset"))
            .withColumn("dataset", F.lit(self._dataset))
            .drop("index")
            .toPandas()
        )

        op_df = op_df.groupby(
            list(set(op_df.columns) - {"attendances", "tele_attendances"}),
            as_index=False,
            dropna=False,
        )[["attendances", "tele_attendances"]].sum()

        op_df["rn"] = op_df.index

        return op_df

    @log_execution
    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe.

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        aae_df = (
            self.spark.read.parquet(f"{self._data_path}/aae")
            .filter(F.col("icb") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumn("sitetret", F.col("dataset"))
            .withColumn("dataset", F.lit(self._dataset))
            .drop("index")
            .toPandas()
        )

        aae_df = aae_df.groupby(
            list(set(aae_df.columns) - {"arrivals"}), as_index=False, dropna=False
        )[["arrivals"]].sum()

        aae_df["rn"] = aae_df.index

        return aae_df

    @log_execution
    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """
        # which year of the ONS population projections to use
        projection_year = 2022
        # load the tables
        births_df = self.spark.read.table(
            "udal_lake_mart.newhospitalprogramme.population_projections_births"
        ).filter(F.col("projection_year") == projection_year)
        catchments_df = self.spark.read.table(
            "udal_lake_mart.newhospitalprogramme.reference_icb_catchments"
        ).filter(F.col("icb") == self._dataset)
        # join and aggregate
        return (
            births_df.join(catchments_df, "area_code")
            .withColumn("sex", F.lit(2))
            .groupBy("age", "sex", F.col("projection").alias("variant"))
            .pivot("year")
            .agg(F.sum(F.col("value") * F.col("pcnt")).alias("value"))
            .toPandas()
        )

    @log_execution
    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        # which year of the ONS population projections to use
        projection_year = 2022
        # load the tables
        demographics_df = self.spark.read.table(
            "udal_lake_mart.newhospitalprogramme.population_projections_demographics"
        ).filter(F.col("projection_year") == projection_year)
        catchments_df = self.spark.read.table(
            "udal_lake_mart.newhospitalprogramme.reference_icb_catchments"
        ).filter(F.col("icb") == self._dataset)
        # join and aggregate
        return (
            demographics_df.join(catchments_df, "area_code")
            .groupBy("age", "sex", F.col("projection").alias("variant"))
            .pivot("year")
            .agg(F.sum(F.col("value") * F.col("pcnt")).alias("value"))
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
                "udal_lake_mart.newhospitalprogramme.default_hsa_activity_tables_icb"
            )
            .filter(F.col("icb") == self._dataset)
            .filter(F.col("fyear") == self._year * 100 + (self._year + 1) % 100)
            .groupBy("hsagrp", "sex", "age")
            .agg(F.mean("activity").alias("activity"))
            .orderBy("hsagrp", "sex", "age")
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
            .filter(F.col("icb") == self._dataset)
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
