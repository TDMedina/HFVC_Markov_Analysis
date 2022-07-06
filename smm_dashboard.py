"""Heart Failure Virtual Clinic - Simple Markov Modelling Dashboard.

@author: T.D. Medina
"""
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, html, dash_table, Output, Input, dcc

from hfvc import processing, modelling

patient_db = processing.main()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "HFVC Markov Modelling"

app.layout = html.Div(children=[
    dbc.NavbarSimple(
        brand="HFVC Markov Modelling",
        brand_href="#",
        color="primary",
        dark=True,
        ),
    # dcc.Slider(id="n_state_slider", min=2, max=10, step=1, value=5),
    dbc.Accordion(always_open=True, children=[
        dbc.AccordionItem(title="Parameters", children=[
            dbc.Row(align="top", justify="start", children=[
                dbc.Col(width="auto", children=[
                    html.Label(dcc.Markdown("**Interval:**"))
                    ]),
                dbc.Col(width="auto", children=[
                    dcc.RadioItems(id="interval_buttons", options=["day", "month", "year"],
                                   labelStyle={"display": "inline-block", "margin-right": "10px"},
                                   value="month")
                    ])
                ]),
            ]),
        dbc.AccordionItem(title="Models by Patient Type", children=[
            html.H2("Models by Patient Type"),
            dcc.Loading(children=html.Div(id="individual_model_tables"))
            ]),
        dbc.AccordionItem(title="Compare Patient Type vs. Others", children=[
            html.H2("Compare Patient Type vs. Others"),
            dcc.Dropdown(list({patient.type for patient in patient_db}),
                         id="patient_type_dropdown",
                         placeholder="Select patient type"),
            dcc.Loading(children=html.Div(id="comparison_model_tables"))
            ])
        ]),
    ])


def make_sd_table(sd_array):
    size = sd_array.shape[0]
    sd_table = pd.DataFrame(data=(sd_array*100).round(1).reshape((1, size)),
                            columns=[f"S{i}" for i in range(1, size+1)],
                            )
    table = dash_table.DataTable(
        data=sd_table.to_dict("records"),
        style_header={"fontWeight": "bold",
                      "backgroundColor": "#b0c4de",
                      "textAlign": "center"}
        )
    return table


def make_tpm_table(tpm_array):
    size = tpm_array.shape[0]
    cols = [f"S{i}" for i in range(1, size+1)]
    tpm_table = pd.DataFrame(data=(tpm_array*100).round(1),
                             columns=cols,
                             )
    tpm_table[""] = cols
    table = dash_table.DataTable(
        data=tpm_table.to_dict("records"),
        columns=[dict(name=x, id=x) for x in [""] + cols],
        style_header={"fontWeight": "bold",
                      "backgroundColor": "#b0c4de",
                      "textAlign": "center"},
        style_cell_conditional=[{"if": {"column_id": ""},
                                 "fontWeight": "bold",
                                 "backgroundColor": "#b0c4de"}],
        )
    return table


def make_model_tables(model, name):
    if not name:
        name = "None"
    shadow_style = "0px 0px 5px 2px rgba(0, 0, 0, 0.2)"
    sd_table = make_sd_table(model[0])
    tpm_table = make_tpm_table(model[1])
    # table_layout = dbc.Col(children=[
    #     html.Label(name),
    #     sd_table,
    #     html.Br(),
    #     tpm_table
    #     ])
    table_layout = dbc.Col(children=[
        dbc.Card(style={"margin": "10px", "box-shadow": shadow_style}, children=[
            dbc.CardHeader(html.B(name)),
            dbc.CardBody(children=[
                html.Label(html.B("Initial Distribution")),
                sd_table,
                html.Br(),
                html.Label(html.B("Transition Probabilities")),
                tpm_table
                ]),
            ])
        ])
    return table_layout


def make_model_table_layout(patient_database, n_states, interval):
    models = modelling.make_all_models(patient_database, n_states, interval)
    row_children = [make_model_tables(model, name) for name, model in models.items()]
    row = dbc.Row(children=row_children)
    return row


def make_comparison_model_layout(patient_database, patient_type, n_states, interval):
    models = modelling.smm_patient_type_against_others(patient_database, patient_type,
                                                       n_states, interval)
    row_children = [make_model_tables(model, name) for name, model in models.items()]
    row = dbc.Row(children=row_children)
    return row


@app.callback(
    Output("individual_model_tables", "children"),
    # Input("n_state_slider", "value"),
    Input("interval_buttons", "value")
    )
# def make_individual_tables(n_states, interval):
#     individual_tables = make_model_table_layout(patient_db, n_states, interval)
def make_individual_tables(interval):
    individual_tables = make_model_table_layout(patient_db, 5, interval)
    return individual_tables


@app.callback(
    Output("comparison_model_tables", "children"),
    Input("patient_type_dropdown", "value"),
    Input("interval_buttons", "value")
    )
def make_comparison_tables(patient_type, interval):
    comparison_tables = make_comparison_model_layout(patient_db, patient_type,
                                                     5, interval)
    return comparison_tables


if __name__ == "__main__":
    app.run_server(debug=True)
