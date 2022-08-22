import logging
import functools
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def log_memory_usage(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start_memory = df.memory_usage().sum()/ 1024**2
        logger.info(f'Memory usage before applying {f.__name__}: {start_memory:5.2f} MB')
        df = f(*args, **kwargs)
        end_memory = df.memory_usage().sum()/ 1024**2
        logger.info(f'Memory usage after: {end_memory:5.2f} MB, {100*(start_memory-end_memory)/start_memory}% reduction')
    return wrapper

@log_memory_usage
def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64',
               # pandas types
                'Int16', 'Int32', 'Int64', 'Float16', 'Float32', 'Float64'
                ]
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).lower()[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np. float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # if col_type == 'object':
    return df


