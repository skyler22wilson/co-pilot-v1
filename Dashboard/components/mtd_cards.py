from Dashboard.setup.common_config import calculate_mtd_metrics, format_currency, format_perc, load_config, calculate_percentage_change
from Dashboard.scripts.create_db import generate_db_file_path
import sqlite3
from dash import html
import dash_bootstrap_components as dbc

def create_mtd_cards():
    db_file_path = generate_db_file_path()
    conn = sqlite3.connect(db_file_path)
    card_contents = calculate_mtd_metrics(conn)


    # Calculating percentage changes
    percentage_changes = {
        'sr': calculate_percentage_change(card_contents['Month-to-Date Sales Revenue'], card_contents['Month-to-Date Sales Revenue Last Year']),
        'gp': calculate_percentage_change(card_contents['Month-to-Date Gross Profit'], card_contents['Month-to-Date Gross Profit Last Year']),
        'gm': calculate_percentage_change(card_contents['Month-to-Date Gross Margin'], card_contents['Month-to-Date Gross Margin Last Year']),
    }

    # Accessing the formatted totals and percentage changes
    total_sales_rev = format_currency(card_contents['Month-to-Date Sales Revenue'])
    gross_profit = format_currency(card_contents['Month-to-Date Gross Profit'])
    gross_margin = format_perc(card_contents['Month-to-Date Gross Margin'])
    perc_change_sr, colour_sr = percentage_changes['sr']
    perc_change_gp, colour_gp = percentage_changes['gp']
    perc_change_gm, colour_gm = percentage_changes['gm']


    sales_revenue_content = html.Div([
        html.H4(total_sales_rev, className="card-title text-center"),
        html.P("Sales Revenue", className="card-text text-center"),
        html.Span(perc_change_sr, className="info-value text-center", style={"color": colour_sr}),
    ])

    gross_profit_content = html.Div([
        html.H4(gross_profit, className="card-title text-center"),
        html.P("Gross Profit", className="card-text text-center"),
        html.Span(perc_change_gp, className="info-value text-center", style={"color": colour_gp}),
    ])

    gross_margin_content = html.Div([
        html.H4(gross_margin, className="card-title text-center"),
        html.P("Gross Margin", className="card-text text-center"),
        html.Span(perc_change_gm, className="info-value text-center", style={"color": colour_gm}),
    ])


    # Combine the KPI contents into a single card with a dynamic title
    return dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(sales_revenue_content, className="dashboard-card"), lg=4, md=12, sm=12, xs=12, className="mb-1"),
                    dbc.Col(dbc.Card(gross_profit_content, className="dashboard-card"), lg=4, md=12, sm=12, xs=12, className="mb-1"),
                    dbc.Col(dbc.Card(gross_margin_content, className="dashboard-card"), lg=4, md=12, sm=12, xs=12, className="mb-1"),
                ],
                className='card-container',
                id='upper-key-metrics'
            ),
        ],
        className="mb-1"  # Add a margin to the bottom if needed
    )