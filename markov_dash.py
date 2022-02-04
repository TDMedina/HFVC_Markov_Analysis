# -*- coding: utf-8 -*-
"""Test Dash App."""

import dash
from dash import dash_table, html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import health_markov


dataset = health_markov.main()
model = health_markov.make_markov_model(
    chains=dataset.make_admission_chains("y", True),
    n_components=5,
    initial_tpm=health_markov.make_random_left_right_matrix(5),
    extend_death_state=1000
    )
tpm = round(pd.DataFrame(model.transmat_), 2)
epm = round(pd.DataFrame(model.emissionprob_), 2)
app = dash.Dash(__name__)


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label("Transition Probability Matrix"),
            dcc.Loading(id="loading-1",
                        type="default",
                        children=html.Div(id="TPM")),
            html.Br(),
            html.Label("Date Resolution"),
            dcc.RadioItems(id="date_resolution_buttons",
                           options=[{"label": "day", "value": "day"},
                                    {"label": "week", "value": "week"},
                                    {"label": "month", "value": "month"},
                                    {"label": "year", "value": "year"}],
                           value="year"),
            html.Br(),
            html.Label("Hidden States"),
            dcc.Slider(id="hidden_states_slider", min=1, max=10, value=5, step=1,
                       marks={str(i): str(i) for i in range(1, 11)},
                       tooltip={"placement": "bottom"}),
            html.Br(),
            html.Label("Observed Death States (10^n)"),
            dcc.Slider(id="death_extender_slider", min=1, max=6, value=3, step=1,
                       marks={str(i): str(i) for i in range(1, 7)},
                       tooltip={"placement": "bottom"}),
            ], style={"padding": 10, "flex": 1}),
        html.Div([
            html.Label("Emission Probability Matrix"),
            dcc.Loading(id="loading-2",
                        type="default",
                        children=html.Div(id="EPM")),
            ], style={"padding": 10, "flex": 1})
        ], style={"display": "flex", "flex-direction": "row"}),
    ])


@app.callback(
    Output("TPM", "children"),
    Output("EPM", "children"),
    Input("date_resolution_buttons", "value"),
    Input("hidden_states_slider", "value"),
    Input("death_extender_slider", "value"),
    )
def update_model(date_res, n_components, death_extension):
    model = health_markov.make_markov_model(
        chains=dataset.make_admission_chains(date_res, True),
        n_components=n_components,
        initial_tpm=health_markov.make_random_left_right_matrix(n_components),
        extend_death_state=10**death_extension
        )
    tpm = round(pd.DataFrame(model.transmat_), 2)
    epm = round(pd.DataFrame(model.emissionprob_), 2)
    tpm = dash_table.DataTable(
        columns=[{"name": f"Stage {i}", "id": str(i)} for i in tpm.columns],
        data=tpm.to_dict("records")
        )
    epm = dash_table.DataTable(
        columns=[{"name": "Not admitted", "id": "0"},
                 {"name": "Admitted", "id": "1"},
                 {"name": "Death", "id": "2"}],
        data=epm.to_dict("records")
        )
    return tpm, epm


if __name__ == "__main__":
    app.run_server(debug=True)
