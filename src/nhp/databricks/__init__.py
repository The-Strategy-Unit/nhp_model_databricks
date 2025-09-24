"""NHP Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from nhp.databricks.national import DatabricksNational
from nhp.databricks.provider import DatabricksProvider
