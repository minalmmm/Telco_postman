from sklearn.model_selection import train_test_split
from preprocessing import *
from sklearn.linear_model import LogisticRegression

def perform_training(data_labelencoded):
    X = data_labelencoded.drop(["Churn"],axis =1 )
    y = data_labelencoded["Churn"]
    X_train , X_test, y_train , y_test = train_test_split(X,y, test_size=0.2)
    X_train , X_test = apply_standardization(X_train , X_test)
    model_log_reg = LogisticRegression()
    model_log_reg.fit(X_train,y_train)
    return X_train , X_test, y_train , y_test, model_log_reg




