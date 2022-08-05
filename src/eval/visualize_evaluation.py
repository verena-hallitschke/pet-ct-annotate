"""
Rendering of evaluation results (see evaluate.py) using dash
"""

import argparse
from datetime import datetime, timezone
import os

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd


def create_evalutaion_app(metrics_df: pd.DataFrame):
    """
    Creates a dash app that displays the metrics in the dataframe

    :param metrics_df: Dataframe that was created by `evaluate.py`containing the study metrics
    :type metrics_df: pd.DataFrame
    :return: Dash app
    :rtype: Dash
    """

    app = Dash("Evaluation Results")

    labels = metrics_df["label"].unique()
    metrics_df["timestamp"] = pd.to_datetime(
        metrics_df["ctime"].apply(
            lambda x: datetime.fromtimestamp(x, tz=timezone.utc).astimezone(tz=None)
        )
    ).dt.strftime("%Y-%m-%d %X")

    @app.callback(
        Output("current-label-name", "data"),
        Output("metrics-dropdown", "options"),
        Output("metrics-dropdown", "value"),
        Output("experiment-checklist", "options"),
        Output("experiment-checklist", "value"),
        Input("tabs-label", "value"),
    )
    def update_tab(selected_label):
        masked_df = metrics_df[metrics_df["label"] == selected_label]
        m_list = masked_df["metric"].unique()
        exp_list = masked_df[masked_df["metric"] == m_list[0]]["experiment"].unique()

        return selected_label, m_list, m_list[0], exp_list, exp_list

    @app.callback(
        Output("experiment-graph", "figure"),
        Input("experiment-checklist", "value"),
        Input("metrics-dropdown", "value"),
        Input("current-label-name", "data"),
    )
    def update_line_chart(experiment_names, current_metric, current_label):

        masked_df = metrics_df[
            (metrics_df.experiment.isin(experiment_names))
            & (metrics_df["metric"] == current_metric)
            & (metrics_df["label"] == current_label)
        ]

        fig = px.line(
            masked_df,
            x="time",
            y="score",
            hover_name="name",
            hover_data=["score", "full_path", "timestamp", "gc"],
            color="experiment",
            markers=True,
        )

        fig.update_layout(xaxis=dict(ticksuffix=" s",))
        return fig

    app.layout = html.Div(
        children=[
            dcc.Store(id="current-label-name"),
            html.H1(children="Evaluation Results"),
            dcc.Tabs(
                id="tabs-label",
                value=f"{labels[0]}",
                children=[
                    dcc.Tab(label=f"{os.path.basename(label)}", value=f"{label}")
                    for label in labels
                ],
            ),
            html.Div([dcc.Graph(id="experiment-graph"),]),
            html.Div(
                [
                    dcc.Checklist(id="experiment-checklist", inline=True),
                    dcc.Dropdown(id="metrics-dropdown", style={"min-width": "150px"}),
                ],
                style={
                    "display": "flex",
                    "gap": "25px",
                    "width": "100%",
                    "align": "left",
                },
            ),
        ]
    )

    return app


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reads a metrics csv file from the given path and renders it using dash."
    )

    parser.add_argument(
        "path", type=str, help="Path to the csv file that should be loaded."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port on which the dash app should be running.",
    )
    parser.add_argument("--debug", action="store_true", help="Activates debug mode.")

    args = parser.parse_args()
    input_data = pd.read_csv(args.path)
    dash_port = args.port

    print(f"Starting the app on 127.0.0.1:{dash_port}")
    create_evalutaion_app(input_data).run_server(debug=args.debug, port=dash_port)
