import logging

from statsforecast.models import _TS
from statsforecast.models import ADIDA
from statsforecast.models import AutoARIMA
from statsforecast.models import AutoETS
from statsforecast.models import AutoTheta
from statsforecast.models import HistoricAverage
from statsforecast.models import IMAPA
from statsforecast.models import Naive
from statsforecast.models import SeasonalNaive
from statsforecast.models import SimpleExponentialSmoothing
from statsforecast.models import Theta
from statsforecast.models import WindowAverage

# from statsforecast.models import MSTL


logger = logging.getLogger(__name__)


class ForecastingModels:
    """A class that contains a dictionary of time series forecasting models."""

    def __init__(self, season_length: int, test_run: bool):
        """Initializes the ForecastingModels class.

        Args:
            season_length (int): number of observations per unit of time
            test_run (bool): if test_run, load only quick heuristics
        """
        self.season_length = season_length
        self.models_dict = {}
        self.test_run = test_run
        logger.info("Loading StatsForecast models")
        self._load_models()

    def _load_models(self) -> None:
        """Loads the forecasting models into the models_dict attribute. If self.test_run, loads only heuristics."""
        if self.test_run:
            self.models_dict = {
                "Naive": Naive(),
                "HistoricAverage": HistoricAverage(),
                "SeasonalNaive": SeasonalNaive(season_length=self.season_length),
                "WindowAverage": WindowAverage(window_size=self.season_length),
            }
            return None

        self.models_dict = {
            "ADIDA": ADIDA(),
            "AutoARIMA": AutoARIMA(season_length=self.season_length),
            "AutoETS": AutoETS(season_length=self.season_length),
            "AutoTheta": AutoTheta(season_length=self.season_length),
            "IMAPA": IMAPA(),
            # "MSTL": MSTL(
            #     season_length=self.season_length
            # ),  # can take Union[int, list[int]]
            "Naive": Naive(),
            "HistoricAverage": HistoricAverage(),
            "SeasonalNaive": SeasonalNaive(season_length=self.season_length),
            "SimpleExponentialSmoothing": SimpleExponentialSmoothing(alpha=0.05),
            "Theta": Theta(season_length=self.season_length),
            "WindowAverage": WindowAverage(window_size=self.season_length),
        }

    def load_fallback_model(self) -> _TS:
        """Returns the fallback forecasting model.

        Returns:
            _TS: fallback model
        """
        return HistoricAverage()

    def get_models_list(self) -> list[_TS]:
        """Returns a list of the forecasting models.

        Returns:
            list[_TS]: list of StatsForecast models
        """
        return list(self.models_dict.values())

    def get_model_names_list(self) -> list[str]:
        """Returns a list of the names of the forecasting models.

        Returns:
            list[str]: names of StatsForecast models
        """
        return list(self.models_dict.keys())
