import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import html
from Dashboard.setup.common_config import load_config
from Dashboard.components.ytd_cards import create_ytd_cards
from Dashboard.components.mtd_cards import create_mtd_cards
from Dashboard.setup.rag_utils import query_output

config = load_config()
table_cols = config['table_cols']
months = config['months']
sales_cols = config['sales_columns']


def register_callbacks(app): 
    
    @app.callback(
        Output('partswise-popover', 'is_open'),
        [Input('partswise-info-button', 'n_clicks')],
        [State('partswise-popover', 'is_open')],
    )
    def toggle_popover(n, is_open):
        if n:
            return not is_open
        return is_open
    
    # Serverside callback for handling the query
    @app.callback(
    [
        Output('partswise-title', 'style'),  # Control the title visibility
        Output('card-logo', 'style'),  # Control the logo visibility
    ],
    [Input('search-button-partswise', 'n_clicks')],
    [State('partswise-input', 'value')]
    )
    def hide_title_and_logo(n_clicks, user_input):
        if n_clicks and user_input.strip():
            # If button clicked and input is not empty, hide the title and logo
            return {'display': 'none'}, {'display': 'none'}
        else:
            # Otherwise, for initial load or empty input, show title and logo
            return {'display': 'block'}, {'display': 'block'}

    @app.callback(
    [
        Output('partswise-output', 'children'),  # Update the output component with query results
        Output('partswise-output', 'style'),  # Adjust style to show or hide the output
        Output('partswise-input', 'value'),  # Clear the input field
        Output('csv-button', 'style')  # Also control the CSV button visibility
    ],
    [Input('search-button-partswise', 'n_clicks')],
    [State('partswise-input', 'value')]
    )
    def handle_query(n_clicks_search, user_input):
        no_display = {'display': 'none'}
        display_block = {'display': 'block'}
        if n_clicks_search > 0 and user_input.strip():
            output_component, has_data = query_output(user_input.strip())  # Assume this returns also whether data is available
            if has_data:
                return output_component, display_block, '', {'display': 'block'}  # Show button if data is available
            else:
                return output_component, display_block, '', no_display
        elif n_clicks_search > 0 and not user_input.strip():
            alert = dbc.Alert("Please enter a query before submitting.", color="warning", style={"marginTop": "15px"})
            return alert, no_display, '', no_display
        return html.Div(), no_display, '', no_display

    # This callback should initiate the CSV download.
    @app.callback(
        Output('table-fig', 'exportDataAsCsv'),
        [Input('csv-button', 'n_clicks')],
        prevent_initial_call=True  # Prevents callback from running on initial load
    )
    def export_data_as_csv(n_clicks):
        if n_clicks and n_clicks > 0:
            # Return the appropriate configuration to AG Grid to trigger the download
            return {'fileName': 'partswise_data.csv'}
        # Since we need to return a dictionary, return an empty one if not triggered
        return {}

    
    @app.callback(
    Output('kpis-output', 'children'),
    [Input('kpis-tabs', 'value')],
    )
    def update_kpis_tab(selected_tab):
        # Extract the database path from the stored data
        if selected_tab == 'tab-ytd':
            return create_ytd_cards()
        elif selected_tab == 'tab-mtd':
            return create_mtd_cards()
        

if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    register_callbacks(app)
    
