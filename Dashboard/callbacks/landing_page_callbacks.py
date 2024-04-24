import dash
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from Dashboard.setup.celery_config import app as celery_app
from Dashboard.setup.celery_config import start_pipeline
from Dashboard.setup.utils import parse_contents
import logging
from celery.result import AsyncResult


logger = logging.basicConfig(level=logging.DEBUG)

MINIMUM_VISIBLE_PROGRESS = 1
PROGRESS_UPDATE_FACTOR = 1
PROGRESS_TEXT_UPDATES = {
    0: 'Starting your build...This may take a minute or two',
    5: 'Processing your data...',
    20: 'Processing complete, moving on to enhancing your data...',
    22: 'Enhancing your data for improved insights into your parts department...',
    30: 'Engineering features for the machine learning model',
    40: 'Machine learning model training...',
    60: 'Categorizing your parts data for in-depth analysis',
    70: 'Calculating the optimal stock level for each part',
    80: 'Applying optimized matrix pricing to maximize gross-profit and increase margins',
    90: 'Enhancement of parts data complete, now building your dashboard...',
    100: 'Dashboard complete!'
}

def landing_page_callbacks(app):

    @app.callback(
    Output("upload-text-container", "children"),  # ID of the container where the text is placed
    [Input('upload-csv', 'filename')]
    )
    def update_upload_text(filename):
        if filename:
            # Return a new Div with the success message
            return html.Div(f"File '{filename}' uploaded. Click submit to start processing.", className="upload-success-text")
        else:
            # Return the original upload prompt
            return html.Div(
                [
                    html.I(className="fas fa-upload fa-2x upload-icon"),
                    html.P("Upload CSV")
                ],
                className="d-flex flex-column align-items-center upload-container"
            )
        
    @app.callback(
    Output('task-store', 'data'),
    [Input('file-submit-button', 'n_clicks')],
    [State('upload-csv', 'contents')]
    )
    def trigger_workflow(n_clicks, contents):
        if n_clicks and contents:
            # Ensure the task is only triggered once per file submission
            parsed_contents = parse_contents(contents)
            task = start_pipeline(parsed_contents)
            return {'task_id': task.id}
        raise PreventUpdate
    
    @app.callback(
    Output('workflow-status', 'data'),
    [Input('progress-interval', 'n_intervals')],
    [State('task-store', 'data')]
    )
    def check_workflow_status(n_intervals, task_data):
        print(f"Checking workflow status at interval {n_intervals} with task data: {task_data}")
        logging.info(f"Checking workflow status at interval {n_intervals} with task data: {task_data}")
        if task_data and 'task_id' in task_data:
            task_id = task_data['task_id']
            task_result = AsyncResult(task_id, app=celery_app)
            if task_result.ready():
                print(f"Task {task_id} completed.")
                logging.info(f"Task {task_id} completed.")
                return {'completed': True}
        print("Workflow not completed or task_id not available.")
        logging.info("Workflow not completed or task_id not available.")
        return {'completed': False}
    
    @app.callback(
    [Output('progress-bar', 'style'),
    Output('progress-bar', 'value'),
    Output('progress-bar', 'label'),
    Output('progress-text', 'className'),
    Output('progress-text', 'children'),
    Output('progress-interval', 'disabled')],
    [Input('file-submit-button', 'n_clicks'), 
     Input("progress-interval", "n_intervals"),
     Input("workflow-status", "data")],  # Note: Input order here
    [State('upload-csv', 'contents')],  # State order here
    prevent_initial_call=True
    )
    def update_progress_visibility(n_clicks, n_intervals, workflow_status, contents):
        if contents is None:
            raise PreventUpdate
        if n_clicks and n_intervals is not None:
            if workflow_status.get('completed', False):
                progress_style = {'display': 'block'}
                text_class_name = 'progress-text'
                progress_text = "Dashboard complete!"
                progress_label = "100%"
                interval_disabled = True

                return progress_style, 100, progress_label, text_class_name, progress_text, interval_disabled

            progress = min(n_intervals % 110, 100)
            #continue only when progress is less than 100
            if progress < 100:
                # Make progress bar and text visible and update accordingly
                progress_style = {'display': 'block'}  # Show progress bar
                text_class_name = 'progress-text'
                progress_text = next((text for val, text in sorted(PROGRESS_TEXT_UPDATES.items(), reverse=True) if progress >= val), "No progress text found.")
                progress_label = f"{progress}%" if progress >= MINIMUM_VISIBLE_PROGRESS else ""
                interval_disabled = False
            else:
                progress_style = {'display': 'block'}
                text_class_name = 'progress-text'
                progress_text = "Dashboard complete!"
                progress_label = "100%"
                interval_disabled = True
            return progress_style, progress, progress_label, text_class_name, progress_text, interval_disabled
        else:
            # Keep progress bar and text hidden
            return {'display': 'none'}, 0, "", 'progress-text-hidden', "", True, False


if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    landing_page_callbacks(app)