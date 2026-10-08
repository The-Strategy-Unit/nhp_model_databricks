import json
import logging
import multiprocessing as mp
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from azure.data.tables import TableServiceClient
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from nhp.model.run import run_all

from nhp.databricks.helpers import DatabricksData

load_dotenv()
logging.basicConfig(level=logging.INFO)
mp.set_start_method("spawn", force=True)


class Run:
    def __init__(
        self,
        params: dict,
        data_class: type[DatabricksData],
        data_path: str,
        save_full_model_results: bool = False,
        **kwargs,
    ):
        create_datetime = datetime.now(tz=UTC)
        params["create_datetime"] = f"{create_datetime:%Y%m%d_%H%M%S}"

        self.params = params
        self._data = data_class(
            spark_builder=self.spark_builder,
            data_path=data_path,
            year=params["start_year"],
            dataset=params["dataset"],
            **kwargs,
        )
        self.save_full_model_results = save_full_model_results
        self.model_run_id = str(uuid.uuid4())

        self._results = None
        self._variants = None
        self._run_metadata = None

    @property
    def metadata(self):
        return {
            k: v
            for k, v in self.params.items()
            if not isinstance(v, dict) and not isinstance(v, list)
        }

    def spark_builder(self):
        cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")

        if not cluster_id:
            raise ValueError("DATABRICKS_CLUSTER_ID environment variable is not set.")
        return DatabricksSession.builder.clusterId(cluster_id).getOrCreate()

    @property
    def data(self):
        return self._data

    @property
    def results(self):
        assert self._results is not None, "Results have not been generated yet."
        return self._results

    @property
    def variants(self):
        assert self._variants is not None, "Variants have not been generated yet."
        return self._variants

    @property
    def run_metadata(self):
        assert self._run_metadata is not None, (
            "Run metadata has not been generated yet."
        )
        return self._run_metadata

    def go(self):
        model_run_start_time = datetime.now(tz=UTC)
        self._results, self._variants = run_all(
            self.params, self.data, save_full_model_results=self.save_full_model_results
        )
        model_run_end_time = datetime.now(tz=UTC)
        elapsed_time = model_run_end_time - model_run_start_time

        self._run_metadata = {
            "model_run_id": self.model_run_id,
            "model_run_start_time": model_run_start_time.isoformat(),
            "model_run_elapsed_time_seconds": elapsed_time.total_seconds(),
            "model_run_end_time": model_run_end_time.isoformat(),
            "viewable": False,
            "status": "complete",
        }

    def upload_results(self):
        results = self.results
        variants = self.variants
        run_metadata = self.run_metadata

        w = WorkspaceClient()
        dbutils = w.dbutils

        secret_scope = "nhp"
        account_name = dbutils.secrets.get(secret_scope, "MLCSU_DATA_ACCOUNT").strip()
        table_name = dbutils.secrets.get(secret_scope, "MLCSU_TABLE_NAME").strip()

        client_id = dbutils.secrets.get(secret_scope, "su-nhp-dev-client-id")
        tenant_id = dbutils.secrets.get(secret_scope, "su-nhp-dev-tenant-id")
        client_secret = dbutils.secrets.get(secret_scope, "su-nhp-dev-client-secret")

        credential = ClientSecretCredential(
            client_id=client_id, tenant_id=tenant_id, client_secret=client_secret
        )

        # upload the parquet files
        blob_client = BlobServiceClient(
            f"https://{account_name}.blob.core.windows.net", credential=credential
        )
        cont = blob_client.get_container_client("results")

        file_path = "/".join(
            [
                "aggregated-model-results",
                self.params["app_version"],
                self.params["dataset"],
                self.params["scenario"],
                self.params["create_datetime"],
            ]
        )

        for k, v in results.items():
            cont.upload_blob(
                file_path + f"/{k}.parquet",
                v.to_parquet(index=False),
                overwrite=True,
                metadata={k: str(v) for k, v in self.metadata.items()},
            )

        # Upload params and variants
        cont.upload_blob(
            f"{file_path}/params.json",
            json.dumps(self.params).encode("utf-8"),
            overwrite=True,
            metadata={k: str(v) for k, v in self.metadata.items()},
        )
        cont.upload_blob(
            f"{file_path}/variants.json",
            json.dumps(variants).encode("utf-8"),
            overwrite=True,
            metadata={k: str(v) for k, v in self.metadata.items()},
        )

        run_metadata["aggregated_results_path"] = file_path

        # Upload the full model results (if required)
        if self.save_full_model_results:
            path = Path(
                f"results/{self.params['dataset']}/{self.params['scenario']}/{self.params['create_datetime']}"
            )
            for file in path.glob("**/*.parquet"):
                filename = file.as_posix()[8:]
                with open(file, "rb") as f:
                    cont.upload_blob(
                        f"full-model-results/{self.params['app_version']}/{filename}",
                        f.read(),
                        overwrite=True,
                    )
                # make sure to remove the file
                os.unlink(file)
            run_metadata["save_full_model_results"] = True
        else:
            run_metadata["save_full_model_results"] = False

        # upload to ATD
        entity = {
            "PartitionKey": self.params["dataset"],
            "RowKey": self.model_run_id,
        } | self.metadata

        entity = entity | run_metadata

        entity["outputs_app_uri"] = f"{self.params['dataset']}/{self.model_run_id}"
        entity["create_datetime"] = self.params["create_datetime"]

        # Upload to ATS
        service_client = TableServiceClient(
            f"https://{account_name}.table.core.windows.net", credential=credential
        )
        table_client = service_client.get_table_client(table_name)

        table_client.create_entity(entity=entity)
