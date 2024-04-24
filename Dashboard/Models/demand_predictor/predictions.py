import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.feature_selection import RFE
import joblib
import logging
import os
import json

CONFIG_FILE = "Dashboard/configuration/SeasonalConfig.json"

# Load configuration
def load_configuration(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)
    
def create_pipeline(X_train, best_params):
    # Initialize the model
    model = XGBRegressor(**best_params)

    # Identify numerical features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('scaler', RobustScaler()),
                ('power_trans', PowerTransformer(method='yeo-johnson'))]),
            numerical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


def preprocess_data(input_data, selected_features):
    print("Loading data...")
    data = json.loads(input_data)  # Correctly load the JSON string into a Python dictionary
    parts_data = pd.DataFrame(data['data'], columns=data['columns'])
    
    # Filter out records with 'months_no_sale' >= 12
    model_data = parts_data[parts_data['months_no_sale'] < 12]

    
    # Select the features (X) and target (y) for modeling
    feature_cols = selected_features
    X = model_data[feature_cols]
    y = model_data['rolling_12_month_sales']
    
    print("Data loaded and preprocessed.")

    # Split the data into training and test sets
    return train_test_split(X, y, test_size=0.33, random_state=42)

def main(current_task, data):
    print('Starting the prediction...')
    model_path = os.path.join('Dashboard', 'Models', 'demand_predictor', 'demand_pipeline.joblib')
    log_path = 'Logs/demand_predictor_logfile.log'

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    config = load_configuration(CONFIG_FILE)
    if not config:
        logging.error('Failed to load configuration.')
        return
    # Configure logging
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info("Demand forecasting script started.")

    try:
        json_path = 'Dashboard/Models/demand_predictor/general_model_details.json'
        with open(json_path, 'r') as file:
            general_model_info = json.load(file)
            # Extract hyperparameters and selected features
            hyperparameters = general_model_info.get('hyperparameters', {})
            selected_features = general_model_info.get('selected_features', [])
    except FileNotFoundError as e:
        logging.error(f"JSON file not found at {json_path}: {e}")
        # Update task state with more specific error message
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'JSON file not found at {json_path}: {e}'})
        return False
    except Exception as e:
        logging.error(f"Error loading JSON file from {json_path}: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error loading JSON file from {json_path}: {str(e)}'})
        return False
    try:  
        
        X_train, X_test, y_train, y_test = preprocess_data(data, selected_features)
        print("Data preprocessing and split completed.")
        

        print('creating pipeline')
        # Create a pipeline with preprocessing and the model
        pipeline = create_pipeline(X_train[selected_features], hyperparameters)
        print('pipeline created')

        pipeline.fit(X_train[selected_features], y_train)
        y_pred = pipeline.predict(X_test[selected_features])
        nan_indices = np.where(np.isnan(y_pred))[0]
        print(f"Number of NaN predictions: {len(nan_indices)}")
        if len(nan_indices) > 0:
            print(f"Indices with NaN predictions: {nan_indices}")
        else:
            print("Predictions:", y_pred)

        # Check if there are NaN values in predictions
        if np.isnan(y_pred).any():
            print("Warning: NaN values found in predictions")

        print("Model fitting completed. Saving model...")
        joblib.dump(pipeline, model_path)
        print(f'Model saved to: {model_path}')

        logging.info("Demand forecasting script completed successfully.")
        return data

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False
