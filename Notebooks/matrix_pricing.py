import pandas as pd
from scipy.optimize import minimize
import logging
import numpy as np
import json

def matrix_pricing(df, upper_quantile=0.9):
    # Error Handling
    try:
        # Filter for 'essential' inventory category
        mask = df['inventory_category'] == 'essential'

        # Calculate upper quartile
        upper_quartile = df[mask]['rolling_12_month_sales'].quantile(upper_quantile)

        # Filter for high-volume parts
        high_volume_df = df[df['rolling_12_month_sales'] >= upper_quartile]

        # Function to optimize the gross profit for a single product
        def optimize_multiplier_for_part(sales_volume, cost, base_price):
            # Determine bounds based on price
            if base_price < 1:
                bounds = (1.0, 10.0)  # Allow larger multipliers for very cheap items
            elif 1 <= base_price < 7.5:
                bounds = (1.0, 3.5)  # Adjust bounds for items priced between $1 and $7.5
            elif 7.5 <= base_price < 10:
                bounds = (1.0, 2.0)  # Adjust bounds for items priced between $7.5 and $10
            elif 10 <= base_price < 15:
                bounds = (1.0, 1.75)  # Adjust bounds for items priced between $10 and $15
            else:
                bounds = (1.0, 1.25)  # Limit multiplier for more expensive items
            
            def gross_profit_function(multiplier):
                price = base_price * multiplier
                gross_profit = (price - cost) * sales_volume
                return -gross_profit  # Negate for minimization
            
            # Optimization
            result = minimize(gross_profit_function, x0=1.0, bounds=[bounds])
            if result.success:
                return result.x[0]
            else:
                logging.warning(f"Optimization failed for part {df['part_number']}")
                return 1.0  # Return default multiplier
        
        # Apply the optimization for each high-volume part
        optimized_prices = []
        for index, row in high_volume_df.iterrows():
            optimized_multiplier = optimize_multiplier_for_part(row['rolling_12_month_sales'], row['cost_per_unit'], row['price'])
            optimized_price = row['price'] * optimized_multiplier
            optimized_prices.append((row['part_number'], optimized_price))

        # Create a DataFrame from the optimized prices
        optimized_prices_df = pd.DataFrame(optimized_prices, columns=['part_number', 'optimized_price'])

        # Merge the optimized prices back into the original DataFrame
        df = df.merge(optimized_prices_df, on='part_number', how='left')

        # Fill in non-optimized parts with original prices
        df['optimized_price'] = np.round(df['optimized_price'].fillna(df['price']), 2)

        return df
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None
    

def main(current_task, input_data):
    try:
        original_data = json.loads(input_data)
        df = pd.DataFrame(original_data['data'], columns=original_data['columns'])
        df.to_feather("/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/data/processed_data/parts_data.feather")
        updated_df = matrix_pricing(df)
        json_data = updated_df.to_json(orient='split')

        return json_data

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False




