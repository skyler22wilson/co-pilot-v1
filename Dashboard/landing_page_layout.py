from dash import dcc, html
import dash_bootstrap_components as dbc

upload_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            html.H2("PartsMatch Co-pilot Data Upload", className="text-center"),
            close_button=False
        ),
        dbc.ModalBody(
            dbc.Container([
                dbc.Row(
                    html.P("Upload Parts Data", className="upload-header-text")
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Upload(
                            id='upload-csv',
                            children=html.Div(
                                [
                                    html.Div(
                                        [
                                            html.I(className="fas fa-upload upload-icon"),
                                            html.P("Upload CSV", className="upload-text")
                                        ],
                                        className="d-flex flex-column align-items-center upload-container",
                                        id="upload-text-container"  
                                    )
                                ],
                                className='upload-area'
                            ),
                            multiple=False  # only single file upload
                        ),
                        width=12,
                    )
                ),
            ], fluid=True),
            style={'borderTop': 'none'},  
        ),
        dbc.ModalFooter(
            dbc.Button("Submit", id="file-submit-button", className="full-width-button", n_clicks=0),
            style={'borderTop': 'none'},  
        ),
        dbc.Row(
                    dbc.Col(
                        [
                            html.Div(id='progress-text', className='progress-bar-hidden'),
                            dcc.Interval(id='progress-interval', interval=1300, n_intervals=0, disabled=True),
                            dbc.Progress(id='progress-bar', className='progress-bar-hidden', value=0, striped=True, animated=True),
                            dcc.Store(id='task-store')
                        ],
                        width=12,
                    )
                )
    ],
    id="upload-modal",
    is_open=True,
    size="md",
    backdrop="static",  # Prevent closing the modal by clicking outside of it
    keyboard=False,  # Prevent closing the modal with the keyboard
    centered=True,  # Center the modal vertically and horizontally
    className='upload-modal d-flex flex-column',  # Flex column to the modal for internal layout
    style={
        'minHeight': '100%',  # Ensures modal content is full height
    },
)


                

