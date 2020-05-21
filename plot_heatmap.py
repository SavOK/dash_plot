from pathlib import Path
import json
import pickle

import pandas as pd
import numpy as np

import plotly.graph_objects as go

_MAIN_DIR = Path("C:/Users/Saveliy/Projects/HHAlign/dash_plot")
_DATA_LOC = _MAIN_DIR / "data_plots"

_PROB_MATRIX = _DATA_LOC / "probability_matrix.txt"


def read_x_names(inFile=_DATA_LOC / "x_group_names.json"):
    with open(inFile) as oF:
        return json.load(oF)


def _nice_lables(grId: int, cut: int = 20, name_dict: dict = None):
    if name_dict is None:
        name_dict = read_x_names()

    name = name_dict[str(griI)]
    if len(name) > cut:
        return f"{name[:cut-3]}..."
    else:
        return name


print("I am here 1 reading the data")

data = pd.read_csv(_PROB_MATRIX, index_col=0)
x_names = read_x_names()
df_x_group = pd.read_csv(
    Path("C:/Users/Saveliy/Projects/HHAlign/data/hhr_data_by_x_group_df.csv"),
    index_col=0
).reset_index(drop=True)

print("I am here 2 Reading the lables")
with open(_DATA_LOC / "nice_names_hover.pckl", "rb") as oF:
    cs_data = pickle.load(oF)

print("I am here 3 Making plot")


fig = go.Figure(
    data=go.Heatmap(
        z=data.iloc[:20, :20],
        x=[i for i in data.columns[:20]],
        y=[i for i in data.index[:20]],
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="HH Probability",
                       font={"size": 20, "family": "Arial"}),
            x=1.04,
        ),

    ),
)

xaxis = dict(
    type="category",
    automargin=True,
    title=dict(text="X Group", font={
               "size": 20, "family": "Arial"}, ),
    # tickvals=data.columns[::20],
    # ticktext=[_nice_lables(i) for i in data.columns[::20]],
)

yaxis = dict(
    type="category",
    automargin=True,
    title=dict(text="X Group", font={
               "size": 20, "family": "Arial"}, standoff=25),
)


# title = {
#     "text": "Maximum HH-probability between X-Groups",
#     "font": {"family": "Arial", "size": 24},
#     "xref": "paper",
#     "xanchor": "center",
#     "x": 0.5,
# }

print("I am here 4 updating layout")
fig.update_layout(
    autosize=True,
    xaxis=xaxis,
    yaxis=yaxis,
    # width=2000,
    # height=2000,
    hovermode="closest",
)

print("I am here 5 saving figgure")
with open(_MAIN_DIR / "saved_plots/heatmap_short.picl", "wb") as oF:
    pickle.dump(fig, oF)
