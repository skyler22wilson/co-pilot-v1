import pandas as pd
import numpy as np
import logging
import json
from joblib import load
from sklearn.calibration import CalibratedClassifierCV

# Load model hyperparameters and preprocessor
def load_model_info(model_info_path, preprocessor_path, model_path):
    with open(model_info_path, 'r') as file:
        model_config = json.load(file)
    preprocessor = load(preprocessor_path)
    model = load(model_path)
    return model_config, preprocessor, model

def get_feature_indices(preprocessor, selected_features):
    """ Map selected feature names to their corresponding indices in the preprocessed array. """
    try:
        # Extract the names of all features in the preprocessed array
        # `transformers_` is a list of tuples representing each transformer
        # Each tuple: (name, transformer_object, feature_names_list)
        all_features = preprocessor.transformers_[0][2]

        # Find indices of each selected feature in the preprocessed feature list
        feature_indices = [list(all_features).index(feature) for feature in selected_features]
        return feature_indices

    except ValueError as e:
        logging.error(f"Error finding feature indices: {e}")
        raise

def calculate_obsolescence_risk(df):
    model_info_path = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/Dashboard/Models/obsolecence_predictor/general_model_details.json'
    preprocessor_path = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/Dashboard/Models/obsolecence_predictor/preprocessor.joblib'
    model_path = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/Dashboard/Models/obsolecence_predictor/calibrated_best_random_forest_model.joblib'

    # Load model configuration, preprocessor, and model
    model_config, preprocessor, model = load_model_info(model_info_path, preprocessor_path, model_path)
    
    selected_features = model_config['selected_features']

    # Filter out obsolete items and assign obsolescence_dummy
    non_obsolete_mask = df['months_no_sale'] < 12
    non_obsolete_df = df[non_obsolete_mask].copy()
    non_obsolete_df['obsolescence_dummy'] = np.where(non_obsolete_df['months_no_sale'] > 6, 1, 0)

    # Preprocess the data and select features
    X_transformed = preprocessor.transform(non_obsolete_df)
    feature_indices = get_feature_indices(preprocessor, selected_features)
    logging.info(f"Feature indicies: {feature_indices}")
    X_selected = X_transformed[:, feature_indices]
    logging.info(f"Selected columns: {X_selected}")

    # Predict probabilities using the calibrated model
    calibrator = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    calibrator.fit(X_selected, non_obsolete_df['obsolescence_dummy'])

    # Predict probabilities using the calibrated model
    predicted_probs = calibrator.predict_proba(X_selected)[:, 1]

    # Assign obsolescence risk to the original DataFrame
    df.loc[non_obsolete_mask, 'obsolescence_risk'] = predicted_probs
    df.loc[~non_obsolete_mask, 'obsolescence_risk'] = 1.0  # or some default value

    return df

def main(current_task, input_data):
    """ Main function to calculate obsolescence risk and handle predictions. """
    # Configure logging
    logging.basicConfig(filename='Logs/obsolescence_predictor.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)

    logging.info('Starting the obsolescence risk calculation process.')
    try:
        original_data = json.loads(input_data)  # Renamed for clarity
        df = pd.DataFrame(original_data['data'], columns=original_data['columns'])
        logging.debug('Data loaded successfully.')

        # Calculate and update obsolescence risk
        logging.debug('Calculating obsolescence risk.')
        df = calculate_obsolescence_risk(df)

        df.to_feather("/Users/skylerwilson/Desktop/PartsWise/Data/Processed/parts_data_obsrisk.feather")
        logging.info(f"Length of obsolescence risk JSON file: {len(df)} rows")

        json_dataset = df.to_json(orient='split')
        logging.info('Obsolescence risk calculation completed successfully.')
        return json_dataset

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        if current_task:
            current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        if current_task:
            current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False


