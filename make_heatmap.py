from pathlib import Path
import pickle

import pandas as pd
import numpy as np

import plotly.graph_objects as go

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

def read_matrix_file(
    inFile: Path = _DATA_LOC / "probability_matrix.txt",
) -> pd.DataFrame:
    return pd.read_csv(inFile, index_col=0)


def read_heatmap_annotation(
    inFile: Path = _DATA_LOC / "heatmap_annotation.pckl",
) -> np.ndarray:
    with open(inFile, "rb") as oF:
        return pickle.load(oF)


print("Reading input data")
data_df = read_matrix_file()
print("Reading heatmap annotation")
heatmap_annotation = read_heatmap_annotation(_DATA_LOC / "heatmap_annotation.pckl")

print("Create heatmap")
heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=data_df.values,
        x=data_df.columns,
        y=data_df.index,
        colorscale="Viridis",
        customdata=heatmap_annotation,
        colorbar=dict(
            title=dict(text="HH Probability", font=dict(size=20, family="Arial"),),
        ),
        hovertemplate="%{customdata[2]}<extra></extra>",
    ),
    layout=go.Layout(
        autosize=True,
        hovermode="closest",
        xaxis=dict(type="category", visible=False, title=dict(text="X group")),
        yaxis=dict(type="category", visible=False, title=dict(text="X group")),
    ),
)

print('Saving heatmap')
with open(_PLOT_LOC / "heatmap_plot.pckl", "wb") as oF:
    pickle.dump(heatmap_fig, oF)