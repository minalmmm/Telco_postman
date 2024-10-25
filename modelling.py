import numpy as np
import pandas as pd
from params import *
from sklearn.preprocessing import LabelEncoder
from preprocessing import *
from training import *
import pickle
from inference import *

def perform_preprocessing_and_return_model_object(data):
    encoder_label = LabelEncoder()
    categorical_columns , numerical_columns = separate_categorical_and_continious_data(data)
    data_labelencoded = encode_categories_with_labelencoder(
                                                            features_list=categorical_columns,
                                                            data = data,
                                                            encoder = encoder_label)
    
    X_train , X_test, y_train , y_test, model_log_reg = perform_training(data_labelencoded)
    with open ("model_.pkl",'wb') as file:
        pickle.dump(model_log_reg,file)
    return X_train , X_test, y_train , y_test, model_log_reg


def return_inference(payload):
    with open(r"./model_.pkl", 'rb') as pickle_file:
        model_ = pickle.load(pickle_file)
    inference_payload = pd.DataFrame(payload)
    predictions = return_predictions(model_, inference_payload)
    return predictions


if __name__ == "__main__":
    #data = pd.read_csv(DATA_PATH)
    
    payload = {
                "customerID":[12345],
                "gender":[0],
                "SeniorCitizen":[0],
                "Partner":[0],
                "Dependents":[0],
                "tenure":[0],
                "PhoneService":[0],
                "MultipleLines":[0],
                "InternetService":[0],
                "OnlineSecurity":[0],
                "OnlineBackup":[0],
                "DeviceProtection":[0],
                "TechSupport":[0],
                "StreamingTV":[0],
                "StreamingMovies":[0],
                "Contract":[1],
                "PaperlessBilling":[0],
                "PaymentMethod":[0],
                "MonthlyCharges":[200.10],
                "TotalCharges":[1000.10]
                }
    X_train , X_test, y_train , y_test, model_log_reg =perform_preprocessing_and_return_model_object(data=  pd.DataFrame(payload))
    return_inference_results = return_inference(payload)
    print(f"return_inference_results: {return_inference_results}")
