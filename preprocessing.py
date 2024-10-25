import numpy as np
import pandas as pd
from params import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def separate_categorical_and_continious_data(data):
    categorical_columns = []
    numerical_columns  =[]
    DataColumns = data.columns
    for col in data[DataColumns]:
        if data[col].dtypes =="object":
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)
    return categorical_columns , numerical_columns


def encode_categories_with_labelencoder(features_list, data, encoder):
    for feature in features_list:
        data[feature] = encoder.fit_transform(data[feature])
    return data

def apply_standardization(X_train , X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test

def apply_standardization_inference(scaler,inference_data):
    inference_data = scaler.transform(inference_data)
    return inference_data


if __name__ == "__main__":
    encoder_label = LabelEncoder()
    data = pd.read_csv(DATA_PATH)
    categorical_columns , numerical_columns = separate_categorical_and_continious_data(data)
    data_labelencoded = encode_categories_with_labelencoder(features_list=categorical_columns,
                                                             data = data,
                                                             encoder = encoder_label)
    print(f"data_labelencoded: {data_labelencoded}")




