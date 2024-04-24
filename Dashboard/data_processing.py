import logging
from setup import db_setup
from scripts import data_clean, monthly_sales, prepare_cols, feature_importances, demand_calc, create_categories, check_reorder, optimal_stock, matrix_pricing
from Models.demand_predictor import demand_predictor
from Models.obsolecence_predictor import obsolete_predictor

# Setup logging
logging.basicConfig(level=logging.INFO)

def process_data():
    logging.info("Data processing started.")
    try:
        data_clean.main()
        monthly_sales.main()
        prepare_cols.main()
        demand_predictor.main()
        feature_importances.main()
        demand_calc.main()
        create_categories.main()
        check_reorder.main()
        obsolete_predictor.main()
        optimal_stock.main()
        matrix_pricing.main()
        logging.info("Preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return

    try:
        # Set up database
        db_setup.main()  # Call the function
        logging.info("Database setup completed successfully.")
    except Exception as e:
        logging.error(f"Error setting up the database: {e}")
        return
    logging.info("Data processing and database setup complete.")

if __name__ == '__main__':
    process_data()



    







