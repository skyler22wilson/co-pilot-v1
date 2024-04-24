from dash import html
import dash_bootstrap_components as dbc

def create_footer():
    return html.Footer(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H4("PartsMatch", className="footer-brand-title"),
                                    html.P("Helping you turn obsolete inventory into working capital for your business "
                                           "with our parts manager co-pilot.", className='footer-slogan'),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-edit"),  # Font Awesome edit icon
                                            " Suggest Edit",
                                        ],
                                        id="suggest-edit-button",
                                        n_clicks=0,
                                        className="ms-auto",
                                    ),
                                    dbc.Modal(
                                    [
                                    html.Div(
                                        [
                                            html.H4("Your Feedback Makes Us Better!", className='custom-modal-header'),
                                            html.P("Help us continuously improve our Co-Pilot. If you see an area that needs refinement, "
                                                "please reach out directly with your suggestions.", className="suggest-edit-header"),
                                            html.Div(
                                                [
                                                    html.I(className="fas fa-envelope contact-icon"),
                                                    html.A("skyler@partsmatch.ai", href="mailto:skyler@partsmatch.ai", className="contact-details")
                                                ], 
                                                className="contact-item"
                                            ),
                                            html.Div(
                                                [
                                                    html.I(className="fas fa-phone contact-icon"),
                                                    html.Span("587-926-7126", className="contact-details")
                                                ], 
                                                className="contact-item"
                                            ),
                                        ],
                                        className='modal-body'
                                    )
                                ],
                                    id="modal-suggest-edit",
                                    is_open=False,
                                    className='modal-dialog'  # Use 'modal-dialog' class for consistent styling with other modals
                                )
                                ],
                                md=4,
                                className="footer-col"
                            ),
                            dbc.Col(
                                [
                                    html.H5("Information", className='footer-text'),
                                    dbc.Row(
                                        html.A(
                                            "About PartsMatch",
                                            href="https://partsmatch.ai/about-partsmatch/",
                                            className="footer-link",
                                            target="_blank",
                                            style={'color': 'gray'}
                                        )
                                    ),
                                    dbc.Row(
                                        html.A(
                                            "Our Story",
                                            href="https://partsmatch.ai/our-story/",
                                            className="footer-link",
                                            target="_blank",
                                            style={'color': 'gray'}
                                        )
                                    ),
                                    dbc.Row(
                                        html.A(
                                            "Join Our Waitlist",
                                            href="https://form.jotform.com/240465727339160",
                                            className="footer-link",
                                            target="_blank",
                                            style={'color': 'gray'}
                                        )
                                    ),
                                    dbc.Row(
                                        html.A(
                                            "Contact Us",
                                            href="https://partsmatch.ai/contact/",
                                            className="footer-link",
                                            target="_blank",
                                            style={'color': 'gray'}
                                        )
                                    ),
                                ],
                                md=4,
                                className="footer-col"
                            ),
                            dbc.Col(
                                [
                                    html.H5("Social Media", className='footer-text'),
                                    html.A(
                                        html.I(className="fab fa-linkedin fa-2x"),
                                        href="https://www.linkedin.com/company/partsmatch/",
                                        className="footer-link",
                                        target="_blank",
                                        style={'color': 'gray'}
                                    ),
                                ],
                                md=4,
                                className="footer-col"
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                "© 2024 PartsMatch. All rights reserved.",
                                className="text-center mt-4",
                            )
                        ]
                    ),
                ],
                fluid=True
            ),
        ],
        className="footer mt-5",
    )



