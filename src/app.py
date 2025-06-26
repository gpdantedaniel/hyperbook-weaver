import tempfile, uuid, pandas as pd, plotly.express as px, gradio as gr
from pathlib import Path
from utils import (
    get_keywords, 
    cluster_keywords, 
    name_clusters, 
    flatten_cluster_map, 
    display_clusters
)

# ---------- Core processing ----------

def pipeline(csv_file):
    """
    A topic modeling pipeline.
    """

    df = pd.read_csv(csv_file.name)
    grouped_keywords = get_keywords(df)
    keywords = list(set([kw for group in grouped_keywords for kw in group]))
    titles = list(df['title'])

    cluster_map = cluster_keywords(keywords)
    cluster_names = name_clusters(cluster_map)
    fig = display_clusters(cluster_map, cluster_names)

    # 3. Produce JSON & HTML artifacts
    tmpdir = Path(tempfile.mkdtemp())
    json_path = tmpdir / f'{uuid.uuid4()}.json'
    html_path = tmpdir / f'{uuid.uuid4()}.html'

    df.to_json(json_path, orient='records', indent=2)
    df.to_html(html_path, index=False)

    # Return expected artifacts
    html_string = html_path.read_text()
    return fig, str(json_path), str(html_path), html_string

with gr.Blocks(title='Hyperbook Weaver 🕷️🦠') as demo:
    gr.Markdown("""# Hyperbook Weaver 🕷️🦠
    Make sure that your .csv dataset has the columns 'title' and 'content'""")

    with gr.Row():
        with gr.Column():
            csv_in = gr.File(label='CSV', file_types=['.csv'])
            run_btn = gr.Button('Process', variant='primary')
        with gr.Column():
            json_out = gr.File(label='Download JSON', file_types=['.json'])
            html_out = gr.File(label='Download HTML', file_types=['.html'])

    plot_out = gr.Plot(label='Scatter plot')
    html_view = gr.HTML(label='HTML preview', show_label=True)

    run_btn.click(pipeline, inputs=csv_in, outputs=[plot_out, json_out, html_out, html_view])

demo.launch()

