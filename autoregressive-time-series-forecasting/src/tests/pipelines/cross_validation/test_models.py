import pytest
from autoregressive_forecasting.pipelines.cross_validation.models import (
    ForecastingModels,
)
from statsforecast.models import _TS


@pytest.fixture
def cv_options():
    cv_options_dict = {
        "h": 1,
        "step_size": 1,
        "n_windows": 1,
        "freq": "D",
        "n_jobs": 1,
        "verbose": False,
        "test_run": True,
    }
    return cv_options_dict


@pytest.fixture
def model_params():
    model_params_dict = {"season_length": 3}
    return model_params_dict


class TestForecastingModels:
    def test_init(self, cv_options, model_params):
        forecasting_models = ForecastingModels(
            **model_params, test_run=cv_options["test_run"]
        )
        forecasting_models._load_models()

        assert isinstance(forecasting_models.models_dict, dict)
        assert len(forecasting_models.models_dict) > 0

    def test_test_run_is_true(self, model_params):
        forecasting_models_true = ForecastingModels(**model_params, test_run=True)
        forecasting_models_true._load_models()
        forecasting_models_false = ForecastingModels(**model_params, test_run=False)
        forecasting_models_false._load_models()

        models_test_run = len(forecasting_models_true.models_dict)
        models_not_test_run = len(forecasting_models_false.models_dict)
        assert models_test_run < models_not_test_run

    def test_load_fallback_model(self, model_params):
        forecasting_models = ForecastingModels(**model_params, test_run=True)
        fallback_model = forecasting_models.load_fallback_model()

        assert isinstance(fallback_model, _TS)

    def test_get_model_list(self, model_params):
        forecasting_models = ForecastingModels(**model_params, test_run=True)
        list_of_models = forecasting_models.get_models_list()

        assert all(isinstance(model, _TS) for model in list_of_models)

    def test_get_model_names_list(self, model_params):
        forecasting_models = ForecastingModels(**model_params, test_run=True)
        list_of_model_names = forecasting_models.get_model_names_list()

        assert all(isinstance(model_name, str) for model_name in list_of_model_names)
