from dash import html, dcc
import dash_bootstrap_components as dbc
from Dashboard.components.partswise_output import get_partswise_card
from Dashboard.scripts.create_db import generate_db_file_path
from Dashboard.components.total_cards import create_total_cards

db_file_path = generate_db_file_path()

kpis_tabs = dcc.Tabs(
    id="kpis-tabs",
    value='tab-ytd',
    children=[
        dcc.Tab(label='Year-To-Date KPIs', value='tab-ytd', className='tab', selected_className='tab--selected'),
        dcc.Tab(label='Month-To-Date KPIs', value='tab-mtd', className='tab', selected_className='tab--selected'),
    ],
    className='custom-tabs'
)

kpis_output = html.Div(id='kpis-output', className='kpis-output')

def create_dashboard():
    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [kpis_tabs, kpis_output],
                                lg=6, md=12, sm=12, xs=12,
                                className="mb-1"
                            ),
                            dbc.Col(
                                create_total_cards(db_file_path), 
                                lg=6, md=12, sm=12, xs=12,
                                className="mb-1"
                            ),
                        ],
                        className='mb-1'
                    ),
                    dbc.Row(
                        [
                            dbc.Col(get_partswise_card(), className='mb-1'),
                        ],
                        className='align-items-stretch partswise-card-row'
                    )
                ],
                fluid=True,
                className="match-height"
            )
        ]
    )



