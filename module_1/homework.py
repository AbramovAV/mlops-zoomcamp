import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

def add_target_variable(df:pd.DataFrame):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    return df

def prepare_dataset(df:pd.DataFrame):
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df.PUlocationID[df.PUlocationID.isna()] = -1
    df.DOlocationID[df.DOlocationID.isna()] = -1

    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].astype(str)

    feature_dicts = df[categorical].to_dict(orient='records')
    target = 'duration'
    y = df[target].values

    print(f"Shape of dataframe: {df.shape}")
    return feature_dicts, y

if __name__=='__main__':
    df_jan = pd.read_parquet('data/fhv_tripdata_2021-01.parquet')
    print(df_jan.describe())
    print(df_jan)
    df_jan = add_target_variable(df_jan)
    print(f"Mean duration of ride in January 2021: {df_jan.duration.mean()}")
    print(f"Fraction of missing values for pickup locations {df_jan.PUlocationID.isna().sum()/df_jan.PUlocationID.isna().size}")
    print(f"Fraction of missing values for dropoff locations {df_jan.DOlocationID.isna().sum()/df_jan.DOlocationID.isna().size}")

    feature_dicts, target = prepare_dataset(df_jan)
    dv = DictVectorizer()
    X_train = dv.fit_transform(feature_dicts)
    y_train = target
    print(f"Shape of feature matrix: {X_train.shape}")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    print(f"RMSE on train: {mean_squared_error(y_train, y_train_pred, squared=False)}")

    df_feb = pd.read_parquet('data/fhv_tripdata_2021-02.parquet')
    print(df_feb.describe())
    df_feb = add_target_variable(df_feb)
    feature_dicts, target = prepare_dataset(df_feb)
    X_val = dv.transform(feature_dicts)
    y_val = target
    y_val_pred = lr.predict(X_val)
    print(f"RMSE on validation: {mean_squared_error(y_val, y_val_pred, squared=False)}")