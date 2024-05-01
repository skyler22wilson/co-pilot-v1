from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import logging
import json

def calculate_obsolescence_risk(df):
    # Load model hyperparameters
    with open('Dashboard/Models/obsolecence_predictor/general_model_details.json', 'r') as file:
        model_config = json.load(file)
    
    selected_features = model_config['selected_features']
    best_hyperparameters = model_config['best_hyperparameters']
    # Filter out obsolete items and assign obsolescence_dummy
    non_obsolete_mask = df['months_no_sale'] < 12
    non_obsolete_df = df[non_obsolete_mask].copy()
    non_obsolete_df['obsolescence_dummy'] = np.where(non_obsolete_df['months_no_sale'] > 6, 1, 0)
    
    X = non_obsolete_df[selected_features]
    y = non_obsolete_df['obsolescence_dummy']
    
    # Preprocessing for numerical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('scaler', RobustScaler()),
            ('power_trans', PowerTransformer(method='yeo-johnson'))]),
         numerical_features)])
    
    # Transform the dataset
    X_transformed = preprocessor.fit_transform(X)

    model = RandomForestClassifier(**best_hyperparameters, random_state=42)
    model.fit(X_transformed, y)

    predicted_probs = model.predict_proba(X_transformed)[:, 1]

    # Assign obsolescence risk to the original dataframe
    df.loc[non_obsolete_mask, 'obsolescence_risk'] = predicted_probs
    # For obsolete parts, you might want to set a default risk or handle differently
    df.loc[~non_obsolete_mask, 'obsolescence_risk'] = 1.0  # or some default value
    
    return df

def main(current_task, input_data):
    # Configure logging
    logging.basicConfig(filename='Logs/obsolescence_predictor.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    
    logging.info('Starting the obsolescence risk calculation process.')
    
    original_data = json.loads(input_data)  # Renamed for clarity
    df = pd.DataFrame(original_data['data'], columns=original_data['columns'])
    logging.debug('Data loaded successfully.')
    try:
        # Calculate and update obsolescence risk
        logging.debug('Calculating obsolescence risk.')
        
        df = calculate_obsolescence_risk(df)
        df.to_feather("/Users/skylerwilson/Desktop/PartsWise/Data/Processed/parts_data_obsrisk.feather")
        json_dataset = df.to_json(orient='split')
        return json_dataset

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False

