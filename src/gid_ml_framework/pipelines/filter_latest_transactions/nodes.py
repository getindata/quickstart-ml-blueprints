import pandas as pd
import logging


logger = logging.getLogger(__name__)

def filter_dataframe_by_last_n_weeks(df: pd.DataFrame, date_column: str, no_weeks: int = 6) -> pd.DataFrame:
    """Filters out records in dataframe older than `max(date) - no_weeks`.

    Args:
        df (pd.DataFrame): dataframe
        date_column (str): name of a column with date
        no_weeks (int, optional): number of weeks to keep. Defaults to 6.

    Returns:
        pd.DataFrame: filtered dataframe
    """
    if not no_weeks:
        logger.info(f'no_weeks is equal to None, skipping the filtering by date step.')
        return df
    try:
        df.loc[:, date_column] = pd.to_datetime(df.loc[:, date_column])
    except KeyError:
        logger.error('Given date_column does not exist in df')
        raise
    except ValueError:
        logger.error('Given date_column is not convertible to datetime')
        raise
    max_date = df.loc[:, date_column].max()
    logger.info(f'Before filtering: dataframe min date: {df[date_column].min()}, max date: {max_date}')
    min_date = max_date - pd.Timedelta(weeks=no_weeks)
    df = df[df[date_column]>min_date]
    logger.info(f'After filtering: dataframe min date: {df[date_column].min()}, max date: {max_date}')
    return df
