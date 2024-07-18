import pandas as pd
import polars as pl
import numpy as np
import joblib
import logging
import shap
import json
from dashboard.setup.utils import load_configuration

# Initialize logging
log_path = 'Logs/demand_predictor_logfile.log'
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

PREPROCESSOR_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/demand_predictor/preprocessor.joblib'
MODEL_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/demand_predictor/xgb_regressor_with_selected_features.joblib'
MODEL_INFO_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/demand_predictor/general_model_details.json'

def preprocess_data(input_data):
    """ Load and preprocess data from JSON input. """
    logging.info("Loading and preprocessing data for demand model...")
    try:
        parts_data = pl.read_json(input_data)
    except Exception as e:
        logging.error(f"Error in processing input data in demand model: {str(e)}")
    model_data = parts_data.filter(pl.col('months_no_sale') < 12)
    return model_data

def get_feature_indices(preprocessor, selected_features):
    """ Map selected feature names to their corresponding indices in the preprocessed array. """
    try:
        all_features = preprocessor.transformers_[0][2]
        return [list(all_features).index(feature) for feature in selected_features]
    except ValueError as e:
        logging.error(f"Error finding feature indices: {e}")
        raise

def calculate_demand(input_data):
    """ Calculate demand score using the pre-trained model and preprocessor. """
    try:
        # Load model info to get selected features
        model_info = load_configuration(MODEL_INFO_PATH)
        selected_features = model_info.get('selected_features', [])
        logging.info(f"Selected features: {selected_features}")
        
        # Load the pre-trained preprocessor and model
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)

        # Preprocess data
        data = json.loads(input_data)
        parts_data = pl.DataFrame(data)
        X = parts_data.select(selected_features)
        
        # Transform the data
        X_transformed = preprocessor.transform(X)

        # Generate predictions
        y_pred = model.predict(X_transformed)

        if np.isnan(y_pred).any():
            logging.warning("NaN values found in predictions")

        logging.info("Predictions generated successfully.")
        
        # Get feature importances
        feature_importances = model.feature_importances_

        # Calculate SHAP values
        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed)

        return y_pred, feature_importances, shap_values

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise



