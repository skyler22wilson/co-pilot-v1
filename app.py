import dash
from dash import html, dcc, Input, Output, State
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
from Dashboard.landing_page_layout import upload_modal 
from Dashboard.dashboard_layout import create_dashboard
from Dashboard.components.get_footer import create_footer
from Dashboard.components.header import create_header
from Dashboard.callbacks.landing_page_callbacks import landing_page_callbacks
from Dashboard.callbacks.dashboard_callback import register_callbacks

app = dash.Dash(__name__, 
                suppress_callback_exceptions=True, 
                external_stylesheets=[
                                        dbc.themes.BOOTSTRAP, 
                                        'https://use.fontawesome.com/releases/v5.15.4/css/all.css', 
                                        'Dashboard/assets'
                                    ]
                )

#header and footer definition
header = create_header()
footer = create_footer()

# Register the callbacks for the dashboard and landing page
landing_page_callbacks(app)
register_callbacks(app)

# Trigger the dashboard content load
@app.callback(
    Output('page-content', 'children'),
    Input('workflow-status', 'data')
)
def trigger_dashboard_navigation(workflow_status):
    print("trigger_dashboard_navigation called with status:", workflow_status)
    if workflow_status.get('completed', False):
        print("Workflow completed")
        return create_dashboard()
    else:
        print("Workflow not completed")
        return dash.no_update

@app.callback(
    Output("modal-suggest-edit", "is_open"),
    [Input("suggest-edit-button", "n_clicks")],
    [State("modal-suggest-edit", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n_clicks, is_open):
    # Toggle the modal open/close based on the button click
    if n_clicks:
        return not is_open
    return is_open

#creates the app
app.layout = html.Div([
    dcc.Store(id='workflow-status', data={'completed': False}),
    header, 
    html.Div(id='page-content', children=[
        upload_modal  # Initial content for the 'page-content' area
    ], style={'flex': '1'}),
    footer 
], style={
    'display': 'flex',
    'flexDirection': 'column',
    'minHeight': '100vh'
})

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)
