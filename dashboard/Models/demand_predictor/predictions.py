import pandas as pd
import polars as pl
import numpy as np
import joblib
import logging
import shap
import json

# Initialize logging
log_path = 'Logs/demand_predictor_logfile.log'
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

MODEL_INFO_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/demand_predictor/general_model_details.json'
FINAL_MODEL_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/demand_predictor/xgb_regressor_with_selected_features.joblib'

def preprocess_data(input_data):
    """ Load and preprocess data from JSON input. """
    logging.info("Loading and preprocessing data for demand model...")
    try:
        if isinstance(input_data, str):
            logging.info(f"Input data received: {input_data[:500]}")  # Log first 500 characters of input data
            parts_data = pl.read_json(input_data)
        elif isinstance(input_data, dict):
            logging.info("Input data received as a dictionary")
            parts_data = pl.DataFrame(input_data)
        else:
            raise ValueError("Invalid input data format")

        model_data = parts_data.filter(pl.col('months_no_sale') < 12)
        return model_data
    except Exception as e:
        logging.error(f"Error in processing input data in demand model: {str(e)}")
        raise

def calculate_shap_values(model, X_transformed_df):
    """ Calculate SHAP values using TreeExplainer for faster performance. """
    try:
        # Use TreeExplainer for tree-based models like XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed_df, approximate=True)
        logging.info("SHAP values calculated successfully using TreeExplainer.")
        return shap_values
    except Exception as e:
        logging.error(f"An error occurred during SHAP value calculation: {e}")
        raise

def calculate_demand(input_data):
    """Calculate demand score using the pre-trained model and preprocessor."""
    try:
        # Load model info to get selected features and best hyperparameters
        with open(MODEL_INFO_PATH, 'r') as file:
            model_info = json.load(file)
        selected_features = model_info['selected_features']
        logging.info(f"Selected features: {selected_features}")
        
        # Load the final pre-trained model pipeline
        logging.info("Loading model pipeline...")
        final_pipeline = joblib.load(FINAL_MODEL_PATH)
        logging.info(f"Model pipeline loaded successfully: {final_pipeline}")
        
        # Preprocess input data
        logging.info("Preprocessing input data...")
        if isinstance(input_data, str):
            logging.info("Input data received as a JSON string")
            data = json.loads(input_data)
        elif isinstance(input_data, dict):
            logging.info("Input data received as a dictionary")
            data = input_data
        else:
            raise ValueError("Invalid input data format")

        parts_data = pd.DataFrame(data)
        logging.info("Input data converted to DataFrame")
        
        # Log available columns in input data
        logging.info(f"Available columns in input data: {parts_data.columns.tolist()}")
        
        # Ensure all necessary features are present in the input data
        missing_features = set(selected_features) - set(parts_data.columns)
        if missing_features:
            raise ValueError(f"Input data is missing required features: {missing_features}")
        
        # Select the necessary features for preprocessing
        X = parts_data[selected_features]
        logging.info(f"Selected features extracted from input data successfully...")
        
        # Transform the data using the preprocessor
        X_transformed = final_pipeline.named_steps['preprocessor'].transform(X)
        logging.info(f"Input data transformed using preprocessor, shape: {X_transformed.shape}, type: {type(X_transformed)}")
        
        # Generate predictions
        y_pred = final_pipeline.named_steps['regressor'].predict(X_transformed)
        logging.info("Predictions generated successfully.")
        
        if np.isnan(y_pred).any():
            logging.warning("NaN values found in predictions")
        
        # Calculate SHAP values
        shap_values = calculate_shap_values(final_pipeline.named_steps['regressor'], X_transformed)
        logging.info("SHAP values calculated successfully.")
        
        return shap_values

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise


