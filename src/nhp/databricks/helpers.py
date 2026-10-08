import logging
import time
from collections.abc import Callable
from functools import wraps

from nhp.model.data import Data
from pyspark.sql import SparkSession


class DatabricksData(Data):
    def __init__(
        self,
        spark_builder: Callable[[], SparkSession],
        data_path: str,
        year: int,
        dataset: str,
    ):
        self._spark_builder = spark_builder
        self._spark = None

        self._data_path = data_path
        self._year = year
        self._dataset = dataset


def log_execution(func):
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        logger.info("starting %s", func.__name__)

        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.exception("failed %s", func.__name__)
            raise
        else:
            elapsed = time.perf_counter() - start
            logger.info("finished %s in (%.1fs)", func.__name__, elapsed)
            return result

    return wrapper
