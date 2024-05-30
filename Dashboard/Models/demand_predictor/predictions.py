import pandas as pd
import numpy as np
import joblib
import logging
import json

# Initialize logging
log_path = 'Logs/demand_predictor_logfile.log'
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

PREPROCESSOR_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/Dashboard/Models/demand_predictor/preprocessor.joblib'
MODEL_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/Dashboard/Models/demand_predictor/xgb_regressor_with_selected_features.joblib'
MODEL_INFO_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/Dashboard/Models/demand_predictor/general_model_details.json'

def load_configuration(config_path):
    """ Load JSON configuration file. """
    try:
        with open(config_path, 'r') as config_file:
            return json.load(config_file)
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise

def preprocess_data(input_data):
    """ Load and preprocess data from JSON input. """
    logging.info("Loading and preprocessing data...")
    data = json.loads(input_data)
    parts_data = pd.DataFrame(data['data'], columns=data['columns'])
    model_data = parts_data[parts_data['months_no_sale'] < 12]
    return model_data

def get_feature_indices(preprocessor, selected_features):
    """ Map selected feature names to their corresponding indices in the preprocessed array. """
    try:
        all_features = preprocessor.transformers_[0][2]
        return [list(all_features).index(feature) for feature in selected_features]
    except ValueError as e:
        logging.error(f"Error finding feature indices: {e}")
        raise

def main(current_task, input_data):
    """ Main function to load model, preprocess data, and make predictions. """
    logging.info("Starting the demand prediction process...")

    try:
        # Load model info to get selected features
        model_info = load_configuration(MODEL_INFO_PATH)
        selected_features = model_info.get('selected_features', [])
        logging.info(f"Selected features: {selected_features}")
        
        # Load the pre-trained preprocessor and model
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)

        # Preprocess data
        X = preprocess_data(input_data)
        
        # Transform the data and select features
        X_transformed = preprocessor.transform(X)
        feature_indices = get_feature_indices(preprocessor, selected_features)
        X_selected = X_transformed[:, feature_indices]

        # Generate predictions
        y_pred = model.predict(X_selected)

        if np.isnan(y_pred).any():
            logging.warning("NaN values found in predictions")

        logging.info("Predictions generated successfully.")
        return input_data

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        if current_task:
            current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': str(e)})
        return False



