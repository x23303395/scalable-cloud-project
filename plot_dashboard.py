import os
import gc
import dask.dataframe as dd
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.express as px
from nltk.corpus import stopwords
import nltk

# Download stopwords if not available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

DATA_DIR = "/home/ubuntu/stream_output_parquet/"

app = Dash(__name__)
app.title = "📚 Real-Time Book Word Trends"

app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', sans-serif",
        "backgroundColor": "#f9f9f9",
        "padding": "30px"
    },
    children=[
        html.H1("📚 Real-Time Book Word Count Dashboard",
                style={"textAlign": "center", "marginBottom": "40px", "color": "#333"}),

        dcc.Interval(
            id="interval-refresh",
            interval=30 * 1000,
            n_intervals=0
        ),

        html.Div([
            dcc.Graph(id="bar-chart"),
        ], style={"marginBottom": "40px", "backgroundColor": "white", "padding": "20px", "borderRadius": "10px",
                  "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.05)"}),

        html.Div([
            dcc.Graph(id="scatter-chart"),
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px",
                  "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.05)"}),

        html.Footer("📅 Auto-refresh every 30s | Real-time data from Kafka + Spark Streaming",
                    style={"textAlign": "center", "marginTop": "40px", "fontStyle": "italic", "color": "#666"})
    ]
)


def load_data():
    try:
        parquet_files = []
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith(".parquet") or file.endswith(".snappy.parquet"):
                    full_path = os.path.join(root, file)
                    if os.path.getsize(full_path) > 0:
                        parquet_files.append(full_path)

        # Only use most recent 5 files to reduce memory usage
        parquet_files = sorted(parquet_files, key=os.path.getmtime, reverse=True)[:5]

        if not parquet_files:
            print("[INFO] No valid parquet files found.")
            return pd.DataFrame(columns=["word", "count"])

        ddf = dd.read_parquet(parquet_files)
        ddf.columns = [str(c).lower() for c in ddf.columns]

        if "token" in ddf.columns and "word" not in ddf.columns:
            ddf = ddf.rename(columns={"token": "word"})

        if "window" in ddf.columns:
            ddf["window_start"] = ddf["window"].apply(lambda w: w.get("start") if isinstance(w, dict) else None,
                                                      meta=("window", "object"))
            ddf["window_end"] = ddf["window"].apply(lambda w: w.get("end") if isinstance(w, dict) else None,
                                                    meta=("window", "object"))
            ddf = ddf.drop(columns=["window"])

        ddf = ddf[~ddf["word"].isin(stop_words)]
        ddf = ddf[ddf["word"].str.len() > 1]

        df = ddf.compute()

        return df

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return pd.DataFrame(columns=["word", "count"])


@app.callback(
    [Output("bar-chart", "figure"),
     Output("scatter-chart", "figure")],
    [Input("interval-refresh", "n_intervals")]
)
def update_dashboard(_):
    df = load_data()

    if df.empty or "word" not in df.columns or "count" not in df.columns:
        no_data_fig = px.bar(title="No data available")
        return no_data_fig, no_data_fig

    top_words = df.sort_values(by="count", ascending=False).head(10)
    bar_fig = px.bar(
        top_words,
        x="word",
        y="count",
        title="🔝 Top 10 Trending Words (Last Few Minutes)",
        labels={"word": "Word", "count": "Frequency"},
        color="count",
        color_continuous_scale="Tealgrn"
    )
    bar_fig.update_layout(
        xaxis_title="Word",
        yaxis_title="Count",
        plot_bgcolor="#fff",
        paper_bgcolor="#fff"
    )

    sample = df.sort_values(by="count", ascending=False).head(100)
    scatter_fig = px.scatter(
        sample,
        x="word",
        y="count",
        size="count",
        color="count",
        title="📌 Word Frequency Distribution (Top 100)",
        labels={"count": "Word Frequency"},
        color_continuous_scale="Turbo"
    )
    scatter_fig.update_layout(
        xaxis_tickangle=45,
        plot_bgcolor="#fff",
        paper_bgcolor="#fff"
    )

    # Free memory
    del df, sample, top_words
    gc.collect()

    return bar_fig, scatter_fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
