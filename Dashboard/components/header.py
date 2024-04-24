import dash_bootstrap_components as dbc
from dash import html

# Define your header layout function or variable
def create_header():
    header = dbc.Navbar(
        dbc.Container(
            [
                # Logo or application name
                dbc.NavbarBrand(
                    [
                        html.Img(
                            src="/assets/logo.png",  # Replace with the path to your logo image
                            height="50px",  # Set the height of the logo, adjust as needed
                        ),
                    ],
                    href="/",  # Link to redirect to when the brand is clicked
                ),
                
                # Navigation or Menu Items (if any)
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Home", href="https://partsmatch.ai")),
                        dbc.NavItem(dbc.NavLink("About", href="https://partsmatch.ai/about-partsmatch/")),
                        dbc.NavItem(dbc.NavLink("Contact Us", href="https://partsmatch.ai/our-story/")),
                        dbc.NavItem(dbc.NavLink("Join Our Waitlist", href="https://form.jotform.com/240465727339160")),
                    ],
                    navbar=True,
                ),
            ],
            fluid=True,  # Set to False for fixed width container
        ),
        className="header mb-4",
    )
    return header