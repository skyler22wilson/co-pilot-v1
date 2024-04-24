from dash import html, dcc
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

logo_path = '/assets/partswise.png'
def get_partswise_card():
    return dbc.Card(
        dbc.CardBody(
            [
                # Container for title, logo, and potentially output
                html.Div(
                    [
                        html.H5(
                            "Your trusted parts department Co-Pilot",
                            className='text-center',
                            id='partswise-title'
                        ),
                        html.Div(
                            html.Img(src=logo_path, className='card-logo'),
                            className='image-container',
                            id='card-logo',
                        ),
                    ],
                    id="title-logo-output-container"
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Download CSV",
                            id="csv-button",
                            n_clicks=0,
                            size="sm",  # Make button small
                            style={'display': 'none'},  # Button starts hidden
                        ),
                        width="auto",
                        className="d-flex justify-content-end"  # Aligns the button to the right
                    ),
                    className="gx-1 my-2"  # Adjusted Bootstrap classes for spacing
                ),
                dcc.Loading(
                    id="loading-1",
                    type="circle",
                    color="#252550",
                    children=[html.Div(id="partswise-output", className="partswise-output")],
                    className='loading-overlay',
                ),
                dcc.Textarea(
                    id='partswise-input',
                    placeholder='Message PartsMatch Co-pilot...',
                    className='d-flex justify-content-center',
                    spellCheck=True
                ),
                html.Div(id='reset-trigger', style={'display': 'none'}),
                html.Button(
                    [html.I(className="fas fa-arrow-up"), " Message PartsMatch Co-pilot"],
                    id='search-button-partswise',
                    n_clicks=0,
                    className='btn-sm custom-button mt-2'
                ),
                html.Div(
                    "Disclaimer: PartsMatch Co-pilot aims to provide accurate information, but it is always recommended to verify the results.",
                    className='text-muted mt-2'
                ),
                html.Button(
                    children=[html.I(className="fas fa-info-circle")],
                    id='partswise-info-button',
                    n_clicks=0,
                    className='info-button btn-info mt-2',
                ),
                dbc.Popover(
                    [
                        dbc.PopoverHeader("How to Use PartsWise AI"),
                        dbc.PopoverBody(
                            "The PartsMatch Co-pilot is an AI tool designed to simplify the management of your parts inventory data. Enter specific questions regarding part numbers, brands, sales data, or obsolescence, and PartsWise will provide accurate and rapid results."
                        ),
                    ],
                    id='partswise-popover',
                    target='partswise-info-button',
                    is_open=False,
                    placement='top'
                ),
            ],
            className='card-body'
        ),
        className='mb-1 dashboard-card',
        id='partswise-card'
    )

def create_table(df):
    # Define the column structure for the AG Grid
    column_defs = [
        {'headerName': col, 'field': col} for col in df.columns
    ]
    rowData = df.to_dict('records')  # Convert DataFrame to a list of dictionaries for rowData

    grid_options = {
         'pagination': True,
         'paginationPageSize': 10,
         'paginationPageSizeSelector': [10, 25, 50, 100, 500, 1000],
         'enablePivot': True,
         'enableSorting': True,
         'enableFilter': True,

    }
    col_def = {"editable": True, "filter": True}
    
    return dag.AgGrid(
        id='table-fig',
        dashGridOptions=grid_options,
        columnDefs=column_defs,
        defaultColDef=col_def,
        rowData=rowData,  # Use rowData here instead of data
        exportDataAsCsv=True,  # Enable CSV export feature
        columnSize="autoSize",
        csvExportParams={
            "fileName": "partswise_data.csv",  # Set the filename for the exported CSV file
        },
    )







