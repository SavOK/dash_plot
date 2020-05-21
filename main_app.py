from pathlib import Path
import json
import pickle

import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html


_MAIN_DIR = Path("C:/Users/Saveliy/Projects/HHAlign/dash_plot")
_DATA_LOC = _MAIN_DIR / "data_plots"
_PLOT_LOC = _MAIN_DIR / "saved_plots"


# loading heatmap json
with open(_PLOT_LOC / "heatmap_short.picl", "rb") as oF:
    heatmap_fig = pickle.load(oF)

# loading x groups names
with open(_DATA_LOC / "x_group_names.json") as oF:
    xgroups = json.load(oF)


def _nice_lables(lb, cut: int = 20):
    if len(lb) > cut:
        return f"{lb[:cut-3]}..."
    else:
        return lb


def _srt_keys(x):
    return int(x[0])


sort_items = sorted(xgroups.items(), key=_srt_keys)
h_group_options = [
    dict(label=f"{k}: {_nice_lables(v, 40)}", value=int(k), title=v)
    for k, v in sort_items
]

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.Div(children=[dcc.Graph(id="heatmap", figure=heatmap_fig)]),
        html.Div(
            children=[
                dcc.Dropdown(
                    id="select-hgroup-1",
                    options=h_group_options,
                    placeholder="Select first X Group",
                ),
            ],
            style=dict(width="48%", display="inline-block"),
        ),
        html.Div(
            children=[
                dcc.Dropdown(
                    id="select-hgroup-2",
                    options=h_group_options,
                    placeholder="Select second X Group",
                ),
            ],
            style=dict(width="48%", display="inline-block"),
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
