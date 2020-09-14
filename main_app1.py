from pathlib import Path
import json
import pickle
from typing import Dict, AnyStr, Tuple

import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd

_OPTIONS = {
    "win_loc": Path("C:\\Users\\Saveliy\\Projects\\HHAlign"),
    "unix_loc": Path("/home/saveliy/HHAlign"),
}

_ENV = "win"
if _ENV == "win":
    _MAIN_LOC = _OPTIONS["win_loc"] / "dash_plot"
else:
    _MAIN_LOC = _OPTIONS["unix_loc"] / "dash_plot"
_DATA_LOC = _MAIN_LOC / "data_plots"
_PLOT_LOC = _MAIN_LOC / "saved_plots"


def read_x_names(
    inFile: Path = _DATA_LOC / "x_group_names.json",
) -> Dict[AnyStr, AnyStr]:
    with open(inFile) as oF:
        return json.load(oF)


def read_matrix_file(
    inFile: Path = _DATA_LOC / "probability_matrix.txt",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0)


def read_x_groupdf(
    inFile: Path = _DATA_LOC / "hhr_data_by_x_group_df.csv",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0).reset_index(drop=True)


def read_main_table(inFile: Path = _DATA_LOC / "hhr_clean.csv") -> pd.DataFrame:
    return pd.read_csv(inFile)


def read_ecod_description_df(
    inFile: Path = _DATA_LOC / "ecod_desc.txt",
) -> pd.DataFrame:
    return pd.read_csv(inFile, sep="|")


def read_heatmap_figure(inFile: Path = _PLOT_LOC / "heatmap_plot.pckl",) -> go.Figure:
    with open(inFile, "rb") as oF:
        return pickle.load(oF)


def get_name_options(name_dict):
    return [{"label": v, "value": k} for k, v in name_dict.items()]


def x_group_label(x_gr: int, name_dict: Dict[AnyStr, AnyStr], cut: int = 20,) -> AnyStr:
    """Convet X group in nice printeble format
    Arguments:
        x_gr {int} -- X group id
        name_dict {Dict[AnyStr, AnyStr]} -- dictionary X group id -> X group name] (default: {names_dict})

    Keyword Arguments:
        cut {int} -- length of the output string (default: {20})

    Returns:
        AnyStr -- name of X group
    """
    name = name_dict[str(x_gr)]
    if len(name) > cut:
        return f"{name[:cut-3]}..."
    else:
        return name


def get_data_histogram_details(x_group: int, main_df: pd.DataFrame) -> pd.DataFrame:
    """ Generates data for histograms details
    Arguments:
        x_group {int} -- X group ID
        main_df {pd.DataFrame} -- dataframe with main data
    Returns:
        pd.DataFrame
    """
    df = main_df[
        (
            (main_df.x_template == main_df.x_query)
            & (main_df.x_template == x_group)
            & (main_df.template != main_df.query)
        )
    ][["template", "query", "prob", "score", "align_ratio"]].reset_index(drop=True)
    df["alignr"] = round(df["align_ratio"], 2)
    return df


def plot_histogram_details(
    xgroup: AnyStr, data: pd.DataFrame, name_dict: Dict
) -> go.Figure:
    """ 
    Plot histogram of details data
    Arguments:
        xgroup {AnyStr} --  x-group id 
        data {pd.DataFrame} -- data to be printed
    Returns:
        go.Figure -- figure
    """
    plot_data = get_data_histogram_details(xgroup, data)
    fig = px.histogram(
        plot_data,
        x="prob",
        marginal="rug",
        hover_data=["query", "template", "prob", "score", "alignr"],
        nbins=10,
        range_x=[5, 105],
    )
    fig.data[0].hovertemplate = "probability = %{x}<br>count = %{y}"
    fig.data[1].hovertemplate = (
        "<b>Domain ID 1 :</b> %{customdata[0]}<br>"
        + "<b>Domain ID 2 :</b> %{customdata[1]}<br>"
        + "<b>Probability :</b> %{customdata[2]}%<br>"
        + "<b>Score :</b> %{customdata[3]}<br>"
        + "<b>Align Ratio :</b> %{customdata[4]}<br>"
    )
    text_label = f"X Group: ({xgroup}) {x_group_label(xgroup, name_dict, 40)}"
    fig.update_layout(
        dict(
            bargap=0.01,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                title={"text": None},
            ),
            xaxis=dict(title="Probability"),
            yaxis=dict(title="Counts"),
            hovermode="closest",
            title=dict(
                text=f"{text_label}", y=0.94, x=0.5, xanchor="center", yanchor="top",
            ),
        )
    )
    return fig


def get_data_histogram_between(id1, id2, main_df):
    df = main_df[((main_df.x_template == id1) & (main_df.x_query == id2))][
        ["template", "query", "prob", "score", "align_ratio"]
    ].reset_index(drop=True)
    df["alignr"] = round(df["align_ratio"], 2)
    return df


def plot_histogram_between(id1, id2, df, name_dict):
    fig = px.histogram(
        df,
        x="prob",
        marginal="rug",
        hover_data=["query", "template", "prob", "score", "alignr"],
        nbins=10,
        range_x=[5, 105],
    )
    fig.data[0].hovertemplate = "probability = %{x}<br>count = %{y}"
    fig.data[1].hovertemplate = (
        "<b>Domain ID 1 :</b> %{customdata[0]}<br>"
        + "<b>Domain ID 2 :</b> %{customdata[1]}<br>"
        + "<b>Probability :</b> %{customdata[2]}%<br>"
        + "<b>Score :</b> %{customdata[3]}<br>"
        + "<b>Align Ratio :</b> %{customdata[4]}<br>"
    )
    text_label = (
        f"X Group: ({id1}) {x_group_label(id1, names_dict, 30)} <br>"
        + f"X Group: ({id2}) {x_group_label(id2, names_dict, 30)}"
    )
    fig.update_layout(
        dict(
            bargap=0.01,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                title={"text": None},
            ),
            xaxis=dict(title="Probability"),
            yaxis=dict(title="Counts"),
            hovermode="closest",
            title=dict(
                text=text_label, y=0.94, x=0.5, xanchor="center", yanchor="top",
            ),
        ),
    )
    return fig


# data for plot
group_df = read_x_groupdf()
main_df = read_main_table()
names_dict = read_x_names()

dropdown_options_labels = get_name_options(names_dict)
# App
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


dropdown_div = html.Div(
    id="select-x-groups",
    children=[
        dcc.Dropdown(
            id="dropdown-select",
            options=dropdown_options_labels,
            placeholder="Select X-group",
            multi=True,
            clearable=False,
        )
    ],
)

app.layout = html.Div(
    id="main-page",
    children=[
        html.H3("Please select two X-groups"),
        html.Div(
            id="main_part_of_page", children=[dropdown_div, html.Div(id="groups-hist")]
        ),
    ],
)


@app.callback(
    Output("main_part_of_page", "children"),
    [Input("dropdown-select", "value")],
    [State("main_part_of_page", "children")],
)
def get_details_hist(value, old_output):
    if value is None:
        raise PreventUpdate
    if len(value) == 1:
        raise PreventUpdate
    if len(value) > 2:
        new_div = html.Div(
            id="main_part_of_page",
            children=[dropdown_div, html.H3("Please choose two X groups")],
        )
        return new_div
    xID1, xID2 = sorted([int(x) for x in value])
    fig_1 = plot_histogram_details(xID1, main_df, names_dict)
    fig_2 = plot_histogram_details(xID2, main_df, names_dict)
    data_3 = get_data_histogram_between(xID1, xID2, main_df)
    if data_3.shape[0] > 0:
        fig_3 = plot_histogram_between(xID1, xID2, data_3, names_dict)
        div_between = html.Div(
            id="right-panel",
            children=[
                dcc.Graph(id="figure3", figure=fig_3, config={"displayModeBar": False},)
            ],
        )
    else:
        div_between = dcc.Markdown(
            children=[
                "No significant alignments between domains from groups  ",
                f"({xID1}) {x_group_label(xID1, names_dict, 40)}   ",
                f"({xID2}) {x_group_label(xID2, names_dict, 40)}   ",
            ],
            style={"textAlign":"center"}
        )
    new_div = html.Div(
        id="main_part_of_page",
        children=[
            dropdown_div,
            html.Div(
                children=[
                    html.Div(
                        id="left-panel",
                        children=[
                            dcc.Graph(
                                id="figure1",
                                figure=fig_1,
                                config={"displayModeBar": False},
                            ),
                        ],
                        style={"width": "50%", "display": "inline-block"},
                    ),
                    html.Div(
                        id="right-panel",
                        children=[
                            dcc.Graph(
                                id="figure2",
                                figure=fig_2,
                                config={"displayModeBar": False},
                            )
                        ],
                        style={"width": "50%", "display": "inline-block"},
                    ),
                ],
                style={"margin": "auto"},
            ),
            div_between,
        ],
    )
    return new_div


if __name__ == "__main__":
    app.run_server(debug=True)
