import dash_bootstrap_components as dbc
from dash import html
from Dashboard.setup.utils import calculate_obsolete_value, calculate_obsolete_percentage
from Dashboard.setup.common_config import calculate_totals_metrics, format_currency,load_config, calculate_percentage_change, turn_result
import sqlite3

def create_total_cards(db_file_path):
    config = load_config()
    conn = sqlite3.connect(db_file_path)

    months = config['months']

    card_contents = calculate_totals_metrics(conn)

    # Formatting currency values
    obsolete_value = format_currency(calculate_obsolete_value(conn))

    # Hardcoded values for last year (for demo purposes)
    industry_standard_turnover = 0.842
    total_cost_last_year = 1310012.61

    # Calculating percentage changes
    percentage_changes = {
        't': calculate_percentage_change(card_contents['12 Month Turnover'], industry_standard_turnover),
        'tc': calculate_percentage_change(card_contents['Total Cost'], total_cost_last_year, decrease_is_positive=True),
        'o': calculate_obsolete_percentage(conn)
    }

    total_cost = format_currency(card_contents['Total Cost'])
    turnover = card_contents['12 Month Turnover']
    perc_change_tc, colour_tc = percentage_changes['tc']
    perc_change_o, colour_o = percentage_changes['o']
    perc_change_t, colour_t = percentage_changes['t']

    comparison_text = turn_result(turnover, industry_standard_turnover)

    total_cost_content = html.Div([
        html.H4(total_cost, className="card-title text-center"),
        html.P("Total Cost of Inventory", className="card-text text-center"),
        html.Span(perc_change_tc, className="info-value text-center", style={"color": colour_tc}),
    ])

    turnover_content = html.Div([
        html.H4(turnover, className="card-title text-center"),
        html.P("12 Month Turnover", className="card-text text-center"),
        html.Span(perc_change_t, className="info-value text-center", style={"color": colour_t}),
        # Uncomment the following line if comparison_text is defined
        #html.Span(comparison_text, className="text-center")
    ])

    obsolescence_content = html.Div([
        html.H4(obsolete_value, className="card-title text-center"),
        html.P("Obsolescence", className="card-text text-center"),
        html.Span(perc_change_o, className="info-value text-center", style={"color": colour_o}),
    ])

    # Combine the sections into a single card
    return dbc.Card(
        [
            html.Div(
                [
                    html.H4("Inventory Totals", className="card-title text-center", style={'font-size': '1.35em', 'margin-top': '8px', 'margin-bottom': '0px'})
                ],
                className="title-container"
            ),
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(dbc.Card(total_cost_content, className="dashboard-card"), lg=4, md=12, sm=12, xs=12, className="mb-1"),
                        dbc.Col(dbc.Card(obsolescence_content, className="dashboard-card"), lg=4, md=12, sm=12, xs=12, className="mb-1"),
                        dbc.Col(dbc.Card(turnover_content, className="dashboard-card"), lg=4, md=12, sm=12, xs=12, className="mb-1"),
                    ],
                    className='card-container',
                    id='upper-key-metrics'
                )
            ),
        ],
        id='parent-card', 
        className="mb-1"
    )
   