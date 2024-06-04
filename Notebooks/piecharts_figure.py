import sqlite3
import pandas as pd
import plotly.graph_objects as go

def create_category(db_file_path):
    # Create a database connection
    conn = sqlite3.connect(db_file_path)
    

    # Combined query to get all required data in one go
    query = """
    SELECT
        inventory_category,
        COUNT(*) AS number_of_parts,
        AVG(months_no_sale) AS avg_months_no_sale,
        SUM(total_cost) AS total_cost,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM parts) AS percentage
    FROM parts
    GROUP BY inventory_category;
    """
    category_data = pd.read_sql_query(query, conn)

    # Prepare hover text using the results directly
    hover_text = [
    f"Category: {row['inventory_category'].replace('_', ' ').title()}<br>Total Cost: ${row['total_cost']:.2f}<br>Avg Months No Sale: {row['avg_months_no_sale']:.2f}<br>Percentage of Inventory:"
    for index, row in category_data.iterrows()
]

    color_map = {
        "essential": "#00FF00",
        "non-essential": "#02CCFE",
        "nearing_obsolete": "#FFA500",
        "obsolete": "red"
    }
    pie_colors = category_data['inventory_category'].map(color_map)

    # Create a pie chart
    fig = go.Figure(data=[go.Pie(
        labels=[label.replace('_', ' ').title() for label in category_data['inventory_category']],
        values=category_data['number_of_parts'],
        marker_colors=pie_colors,
        text=hover_text,
        hoverinfo='text+percent',
        textinfo='percent',
        hole=.3,  # Adding a hole to make it a donut chart
    )])

    # Styling the chart
    fig.update_layout(
        title={
            'text': "Percentage of Parts by Inventory Category",
            'y':1.0,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
            },
        title_font=dict(family="Cabin, sans-serif", size=20, color="#444444"),
        legend=dict(
            font=dict(family="Cabin, sans-serif", size=20, color="#444444"),
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        margin=dict(l=0, r=0, t=50, b=60),
    )
    # Customizing the hover label
    fig.update_traces(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Cabin, sans-serif"))

    return fig

