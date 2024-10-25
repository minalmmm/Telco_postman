from sklearn.linear_model import LogisticRegression
from preprocessing import *
from sklearn.preprocessing import LabelBinarizer

def return_predictions(model_,infernece_payload):
    encoder_label = LabelEncoder()
    categorical_columns , numerical_columns = separate_categorical_and_continious_data(infernece_payload)
    infernece_payload = encode_categories_with_labelencoder(
                                                            features_list=categorical_columns,
                                                            data = infernece_payload,
                                                            encoder = encoder_label
                                                            )
    
    # infernece_payload = apply_standardization_inference(scaler,infernece_payload)
    model_log_reg_predictions = model_.predict(infernece_payload)
    return model_log_reg_predictions
