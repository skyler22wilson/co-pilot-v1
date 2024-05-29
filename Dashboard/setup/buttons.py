from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd

def create_buttons():

    html.Div([
        html.Button("MTD Sales", id='btn-mtd-sales', n_clicks=0),
        html.Button("YTD Profits", id='btn-ytd-profits', n_clicks=0),
        html.Div(id='output-container')
    ])