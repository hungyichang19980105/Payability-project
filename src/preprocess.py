# Import Libraries
import hydra
import numpy as np
import pandas as pd
import pathlib
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath


@hydra.main(config_path="config", config_name='main')
def preprocess_data(config: DictConfig):

    # get current path
    current_path = hydra.utils.get_original_cwd() + "/"
    path = current_path + config.raw.path

    # Read data
    df = pd.read_csv(path, dtype=config.raw.type)
    df = df.drop(df.columns[[0]], axis=1)
    df = df.drop('acceptor_state', axis=1)
    df = df.drop('pin_present', axis=1)
    # df.add

    # Convert amount to float
    def s2f(s):
        return float(str(s).replace(",", ""))
    df['cc_amount'] = df['cc_amount'].apply(s2f)

    # Add new features of time derived variables
    df['user_transaction_time'] = pd.to_datetime(
        df['user_transaction_time'], errors='coerce', utc=True)
    df['hour'] = df['user_transaction_time'].dt.hour
    df['month'] = df['user_transaction_time'].dt.month
    df['dayofweek'] = df['user_transaction_time'].dt.dayofweek
    df['year'] = df['user_transaction_time'].dt.year

    # Remove user_transaction time and date
    df = df.drop(['user_transaction_time', 'date'], axis=1)

    # Convert categorical variable's type to category
    cat_vars = config.variables.cat_num_vars
    for col in cat_vars:
        df[col] = df[col].fillna(-1)
        df[col] = df[col].astype(float)
        df[col] = df[col].astype(int)
        df[col] = df[col].astype(str)
        df[col] = df[col].replace('-1', np.nan)

    # Export data
    df.to_csv(current_path + config.processed.path)


if __name__ == '__main__':
    preprocess_data()
