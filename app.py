import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
import json  # Import json for data serialization

from params import *  
from sklearn.preprocessing import LabelEncoder
from preprocessing import *  
from training import *  
import pickle
from inference import *  
from modelling import *  

app = Flask(__name__)

# Database connection
DATABASE_URL = r'sqlite:///C:\data science material\telco_churn\notebook\data\my_database.db'
engine = create_engine(DATABASE_URL)

def save_to_database(data, table_name):
    """ Save DataFrame to SQL database. """
    try:
        data.to_sql(table_name, con=engine, if_exists='append', index=False)
    except Exception as e:
        print(f"Error saving to database: {str(e)}")

# Health Check Endpoint
@app.route("/health_check", methods=['GET'])
def get_health_check():
    """ Health Check Endpoint to check if the API is alive. """
    response = {"response_status": 200, "response_body": "Alive"}
    return jsonify(response), 200

# Training Endpoint
@app.route("/training", methods=['POST'])
def start_training():
    """ Endpoint to handle model training with uploaded data. """
    if 'file' not in request.files:
        return jsonify({"Error_Code": 400, "Error_Message": "No File has been passed"}), 400
    
    file = request.files['file']
    
    try:
        # Read and preview the uploaded CSV file
        training_data = pd.read_csv(file)
        print("File successfully read, here is the preview:")
        print(training_data.head())
        
        # Save training data to SQL database
        save_to_database(training_data, 'training_data')  # Table name
        
        # Perform preprocessing and model training
        X_train, X_test, y_train, y_test, model_log_reg = perform_preprocessing_and_return_model_object(data=training_data)
        
        return jsonify({
            "Status_Code": 200,
            "Message": "Training data received and processed successfully.",
            "Sample_Data": training_data.head(5).to_dict(orient='records')
        }), 200
    
    except pd.errors.EmptyDataError:
        return jsonify({"Error_Code": 400, "Error_Message": "Uploaded file is empty."}), 400
    except pd.errors.ParserError:
        return jsonify({"Error_Code": 400, "Error_Message": "Error parsing the CSV file."}), 400
    except Exception as e:
        return jsonify({"Error_Code": 400, "Error_Message": f"File Processing Error: {str(e)}"}), 400

# Inference Endpoint
@app.route("/inference", methods=['POST'])
def get_inference_results():
    """ Endpoint to get inference results from the model. """
    try:
        # Get JSON payload from the request
        payload = request.get_json()
        
        # Check if payload is empty
        if not payload:
            return jsonify({"Error_Code": 400, "Error_Message": "No input data provided."}), 400
        
        # Perform inference using the payload
        inference_result = return_inference(payload)

        # Convert payload to JSON string for storage
        input_data_json = json.dumps(payload)

        # Save results to SQL database
        results = pd.DataFrame({
            'input_data': [input_data_json],  # Save the input data as a JSON string
            'prediction': [str(inference_result[0])]  # Save the prediction
        })
        save_to_database(results, 'predictions')  # Save to 'predictions' table
        
        # Return the result
        return jsonify({"Status": "Success", "Target": str(inference_result[0])}), 200
    
    except Exception as e:
        return jsonify({"Error_Code": 400, "Error_Message": f"Inference Error: {str(e)}"}), 400

# Download Results Endpoint
@app.route("/download_results", methods=['GET'])
def download_results():
    """ Endpoint to download results from the database as a CSV file. """
    try:
        # Load results from the predictions table
        query = "SELECT * FROM predictions;"
        results_df = pd.read_sql(query, con=engine)

        # Save DataFrame to CSV
        results_df.to_csv('predictions.csv', index=False)

        # Return a success message
        return jsonify({"Status": "Success", "Message": "Results saved to predictions.csv"}), 200

    except Exception as e:
        return jsonify({"Error_Code": 500, "Error_Message": f"Error downloading results: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)



