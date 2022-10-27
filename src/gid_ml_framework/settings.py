"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

# flake8: noqa
# # Instantiated project hooks.
# from gid_ml_framework.hooks import ProjectHooks
# HOOKS = (ProjectHooks(), )

DISABLE_HOOKS_FOR_PLUGINS = ("kedro-mlflow",)
from kedro_vertexai.auth.mlflow_request_header_provider_hook import MLFlowRequestHeaderProviderHook
from kedro_vertexai.auth.gcp import MLFlowGoogleIAMRequestHeaderProvider
from kedro_mlflow.framework.hooks import MlflowHook
from gid_ml_framework.hooks import MemoryProfilingHooks, IgnoreDeprecationWarnings
HOOKS = (
    MlflowHook(),
    MLFlowRequestHeaderProviderHook(MLFlowGoogleIAMRequestHeaderProvider),
    MemoryProfilingHooks(),
    IgnoreDeprecationWarnings()
    )

# # Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import ShelveStore
# SESSION_STORE_CLASS = ShelveStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# # Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
# from kedro.config import TemplatedConfigLoader
# CONFIG_LOADER_CLASS = TemplatedConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
# CONFIG_LOADER_ARGS = {
#     "globals_pattern": "*globals.yml",
# }

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
