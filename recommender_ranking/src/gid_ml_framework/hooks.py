"""Project hooks."""
import warnings
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog

from gid_ml_framework.helpers.utils import reduce_memory_usage


class ProjectHooks:
    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str], env: str, extra_params: Dict[str, Any]
    ) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version
        )


class MemoryProfilingHooks:
    @hook_impl
    def after_dataset_loaded(self, dataset_name: str, data: Any) -> None:
        if isinstance(data, pd.DataFrame):
            data = reduce_memory_usage(data)


class IgnoreDeprecationWarnings:
    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
