import pandas as pd
import mlflow
from pandas_profiling import ProfileReport
import logging

log = logging.getLogger(__name__)

def auto_eda(df: pd.DataFrame, name: str) -> None:
    """Automatic exploratory data analysis from pandas_profiling. Saves results
    into mlflow.

    Args:
        df: dataframe on which to run auto EDA
    """
    minimal = True if df.size > 1_000_000 else False
    log.info(f'Setting {minimal=} for dataframe: {name}')
    profile = ProfileReport(df, title=f"Pandas Profiling Report - {name}", minimal=minimal)
    profile_html = profile.to_html()
    mlflow.log_text(profile_html, f"eda/aut_eda_{name}.html")
