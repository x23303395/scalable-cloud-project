import os
import pandas as pd
import dask.dataframe as dd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import time

SENTIMENT_DIR = "/home/ubuntu/sentiment_output_parquet/"
MAX_FILES = 50  # Limit for speed

app = Dash(__name__)
app.title = "📖 Real-Time Book Sentiment Dashboard"

app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', sans-serif",
        "backgroundColor": "#f9f9f9",
        "padding": "30px"
    },
    children=[
        html.H1(
            "📉 Real-Time Book Sentiment Analysis",
            style={"textAlign": "center", "color": "#222", "marginBottom": "40px"}
        ),

        dcc.Interval(
            id="interval-refresh",
            interval=30 * 1000,
            n_intervals=0
        ),

        html.Div([
            dcc.Loading(dcc.Graph(id="line-chart")),
        ], style={"marginBottom": "40px", "backgroundColor": "white", "padding": "20px", "borderRadius": "10px",
                  "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.05)"}),

        html.Div([
            dcc.Loading(dcc.Graph(id="bar-chart")),
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px",
                  "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.05)"}),

        html.Footer("🕒 Auto-refresh every 30s | Data Flow: Kafka → Spark → Parquet → Dashboard",
                    style={"textAlign": "center", "marginTop": "40px", "fontStyle": "italic", "color": "#666"})
    ]
)


def get_sentiment_label(score):
    if score >= 0.6:
        return "😊 Very Positive"
    elif score >= 0.3:
        return "🙂 Positive"
    elif score >= -0.3:
        return "😐 Neutral"
    elif score >= -0.6:
        return "🙁 Negative"
    else:
        return "😠 Very Negative"


def load_sentiment_data():
    start = time.time()
    try:
        all_files = []
        for root, dirs, files in os.walk(SENTIMENT_DIR):
            if "_temporary" in root:
                continue
            for file in files:
                if file.endswith(".parquet") or file.endswith(".snappy.parquet"):
                    full_path = os.path.join(root, file)
                    if os.path.getsize(full_path) > 0:
                        all_files.append((full_path, os.path.getmtime(full_path)))

        if not all_files:
            print("[INFO] No non-empty parquet files found.")
            return pd.DataFrame(columns=["book", "avg_sentiment", "window_start"])

        # Sort by modified time (descending) and pick latest MAX_FILES
        sorted_files = sorted(all_files, key=lambda x: x[1], reverse=True)
        latest_files = [f[0] for f in sorted_files[:MAX_FILES]]

        try:
            ddf = dd.read_parquet(latest_files)
            df = ddf.compute()
        except Exception as e:
            print(f"[ERROR] Dask failed, trying fallback with pandas. Reason: {e}")
            frames = []
            for f in latest_files:
                try:
                    frame = pd.read_parquet(f)
                    frames.append(frame)
                except Exception as err:
                    print(f"[WARNING] Skipped corrupt file: {f}, reason: {err}")
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]

        if "window" in df.columns:
            df["window_start"] = df["window"].apply(lambda w: w.get("start") if isinstance(w, dict) else None)
            df.drop(columns=["window"], inplace=True)

        df = df.dropna(subset=["book", "avg_sentiment", "window_start"])
        print(f"[INFO] Loaded {len(df)} rows in {time.time() - start:.2f}s")
        return df

    except Exception as e:
        print(f"[ERROR] load_sentiment_data failed: {e}")
        return pd.DataFrame(columns=["book", "avg_sentiment", "window_start"])


@app.callback(
    [Output("line-chart", "figure"),
     Output("bar-chart", "figure")],
    [Input("interval-refresh", "n_intervals")]
)
def update_charts(_):
    df = load_sentiment_data()

    if df.empty or "window_start" not in df.columns:
        dummy = pd.DataFrame({
            "book": ["No Data"],
            "avg_sentiment": [0],
            "window_start": [pd.Timestamp.now()]
        })
        fig = px.line(dummy, x="window_start", y="avg_sentiment", title="No sentiment data available")
        return fig, fig

    # Line Chart
    line_fig = px.line(df, x="window_start", y="avg_sentiment", color="book",
                       title="📈 Sentiment Over Time (per Book)",
                       labels={"avg_sentiment": "Sentiment Score", "window_start": "Time"})

    # Bar Chart - latest window only
    latest_time = df["window_start"].max()
    latest_df = df[df["window_start"] == latest_time].copy()
    latest_df["sentiment_label"] = latest_df["avg_sentiment"].apply(get_sentiment_label)

    color_map = {
        "😊 Very Positive": "#43a047",
        "🙂 Positive": "#66bb6a",
        "😐 Neutral": "#fbc02d",
        "🙁 Negative": "#ef6c00",
        "😠 Very Negative": "#d32f2f"
    }

    bar_fig = px.bar(
        latest_df,
        x="book",
        y="avg_sentiment",
        color="sentiment_label",
        title="📚 Latest Sentiment Scores per Book",
        labels={"avg_sentiment": "Sentiment Score", "book": "Book"},
        custom_data=["sentiment_label"]
    )

    bar_fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<br>Sentiment: %{customdata[0]}",
        marker_color=latest_df["sentiment_label"].map(color_map)
    )

    bar_fig.update_layout(xaxis_tickangle=45)

    return line_fig, bar_fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

