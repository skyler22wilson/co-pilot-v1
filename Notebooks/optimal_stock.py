import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import json

def run_model(data):
    
    features = ['demand', 'total_cost', 'obsolescence_risk', 'roi', 'rolling_12_month_sales']
    scaler = MinMaxScaler()

    # Scale features and extract them as tensors
    scaled_features = scaler.fit_transform(data[features])
    tensors = [tf.constant(scaled_features[:, i], dtype=tf.float32) for i in range(scaled_features.shape[1])]
    demand, total_cost, obsolescence_risk, roi, sales = tensors

    # Adjust quantities based on normalized sales
    normalized_sales = sales * demand
    adjusted_quantities = np.where(data['quantity'] == 0, normalized_sales, data['quantity'])
    quantities = tf.Variable(adjusted_quantities, dtype=tf.float32)

    # Define optimizer with a learning rate schedule
    initial_learning_rate = 0.025
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=350,
        decay_rate=0.95,
        staircase=False
    )
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

    # Define weights and training step
    weights = {
            'w_tc': tf.Variable(0.5, dtype=tf.float32),
            'w_or': tf.Variable(0.35, dtype=tf.float32),
            'w_r': tf.Variable(-0.75, dtype=tf.float32),
        }

    def train_step():
        with tf.GradientTape() as tape:
            # Calculate each component with appropriate signs
            # Costs and risks increase the loss
            cost_val = tf.reduce_sum(weights['w_tc'] * quantities * total_cost)  # weight should be positive
            risk_val = tf.reduce_sum(weights['w_or'] * quantities * obsolescence_risk)  
            roi_val = tf.reduce_sum(weights['w_r'] * quantities * roi)  

            # Formulate the loss with signs correctly assigned
            loss = (cost_val + risk_val) - (roi_val)

        gradients = tape.gradient(loss, [quantities])
        optimizer.apply_gradients(zip(gradients, [quantities]))
        quantities.assign(tf.maximum(quantities, 0))  # Ensure non-negativity
        return loss.numpy()

    def apply_constraints(quantities, inventory_data):
        quantities_np = quantities.numpy()
        
        # Additional constraints based on inventory category
        essential_mask = inventory_data['inventory_category'] == 'essential'
        obsolete_mask = inventory_data['inventory_category'] == 'obsolete'

        # Adjust for essential and obsolete items
        for i, (is_essential, is_obsolete) in enumerate(zip(essential_mask, obsolete_mask)):
            if is_essential:
                min_quantity = max(1, inventory_data['quantity'][i])  # Ensure a minimum quantity for essential items
                quantities_np[i] = max(quantities_np[i], min_quantity)
            elif is_obsolete:
                quantities_np[i] = 0  # Set obsolete item quantities to zero

        # Update TensorFlow variable
        quantities.assign(quantities_np)

        # Optimization loop
    epochs = 500
    for _ in range(epochs):
        train_step()
        apply_constraints(quantities, data)

    updated_quantities_np = quantities.numpy()

    # Update DataFrame with optimized quantities
    data['optimized_quantity'] = updated_quantities_np.astype(int)

    return data

def determine_inventory_status(row):
    if row['inventory_category'] == 'obsolete':
        return 'do_not_reorder'
    if row['optimized_quantity'] < row['quantity']:
        return 'overstocked'
    elif row['optimized_quantity'] > row['quantity']:
        return 'understocked'
    elif row['optimized_quantity'] == row['quantity']:
        return 'optimally_stocked'
    elif row['quantity'] == row['reorder_point']:
        return 'reorder'
    else:
        return 'check'  # This is a fallback case if none of the above conditions are met


def main(current_task, input_data):
    # Load and prepare data
    try:
        original_data = json.loads(input_data)
        df = pd.DataFrame(original_data['data'], columns=original_data['columns'])
        logging.info(f"Columns in dataset before optimal stock: {df.columns}")
        # This is your main entry point for the script
        result = run_model(df)
        result['inventory_status'] = result.apply(determine_inventory_status, axis=1)

        json_data = result.to_json(orient='split')

        return json_data

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False
