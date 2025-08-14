import os
import pandas as pd
import gradio as gr
from utils import TopicWeaver

available_models = {
    'azure': {
        'name': 'Azure OpenAI models',
        'models': (
            'text-embedding-3-large',
            'text-embedding-3-small',
            'text-embedding-ada-002'
        ),
    },
    'hf inference': {
        'name': 'HuggingFace models',
        'models': (
            'Qwen/Qwen3-Embedding-0.6B',
            'Qwen/Qwen3-Embedding-4B',
            'Qwen/Qwen3-Embedding-8B',
            'BAAI/bge-large-en-v1.5',
            'intfloat/multilingual-e5-large-instruct'
        )
    },
    'ollama': {
        'name': 'Ollama models',
        'models': (
            'hf.co/Qwen/Qwen3-Embedding-4B-GGUF:F16',
            'dengcao/Qwen3-Embedding-0.6B:Q8_0',
            'dengcao/Qwen3-Embedding-0.6B:F16',
            'dengcao/Qwen3-Embedding-4B:Q4_K_M',
            'dengcao/Qwen3-Embedding-8B:Q4_K_M',
            'dengcao/Qwen3-Embedding-8B:Q8_0'
        )
    }
}

weaver = None

def pipeline(csv_file, model_name, provider, cluster_min, top_n, chunk_size, naming_method):
    """
    A convenient pipeline that orchestrates all the steps
    involved in topic modeling, from keyword extraction to the
    generation of an interactive D3S.js graph of the articles.
    """

    # Extract content
    df = pd.read_csv(csv_file.name)
    docs, titles = df['content'], df['title']
    filename = os.path.basename(csv_file.name)

    # Initialize progress bar and the Hyperbook Weaver
    progress = gr.Progress() 
    weaver = TopicWeaver(
        model_name=model_name, 
        inference_mode=provider, 
        top_n=top_n, 
        cluster_min=cluster_min, 
        chunk_size=chunk_size,
        naming_method=naming_method
    )

    # Extract keywords from the documents
    docs_keywords = [weaver.get_keywords(doc) for doc in progress.tqdm(docs, desc='Extracting keywords')]

    # Step 2. Embed and cluster the unique keywords
    unique_keywords = list(set([kw for keywords in docs_keywords for kw in keywords]))
    cluster_map, keyword_embeddings = weaver.cluster_keywords(unique_keywords)

    # Step 3. Find names for every cluster
    cluster_names = {
        cluster_id: weaver.name_cluster(keywords, keyword_embeddings, naming_method) 
        for cluster_id, keywords in progress.tqdm(cluster_map.items(), desc='Naming clusters')
    }

    # Step 4. Display the clusters
    fig2d = weaver.display_2d(cluster_map, cluster_names, keyword_embeddings)
    fig3d = weaver.display_3d(cluster_map, cluster_names, keyword_embeddings)

    del keyword_embeddings  # Free up some GPU memory

    # Step 5. Generate the graph and its HTML display
    G, kw2cluster = weaver.create_graph(cluster_map, cluster_names, titles, docs_keywords)
    html_graph = weaver.display_graph(G, filename)

    # Step 6. Give each article its tag
    tags_df, tags_path = weaver.assign_tags(titles, docs_keywords, kw2cluster, cluster_names, filename)

    return fig2d, fig3d, html_graph, tags_df, tags_path

def update_models(provider):
    return gr.Dropdown(
        choices=available_models[provider]['models'], 
        value=available_models[provider]['models'][0],
        label=available_models[provider]['name'],
        interactive=True,
        allow_custom_value=True
    )

def update_ngram(ngram_lower, ngram_higher):

    ngram_lower = gr.Number(label='Lower end', interactive=True, minimum=1, maximum=ngram_higher)
    ngram_higher = gr.Number(label='Higher end', interactive=True, minimum=ngram_lower)

    return ngram_lower, ngram_higher
    
with gr.Blocks(title='Hyperbook Weaver 🕷️') as demo:

    gr.Markdown('# 🕷️ Hyperbook Weaver')

    with gr.Tab(label='Topic plots'):
        plot_2d = gr.Plot(label='2D topic plot')
        plot_3d = gr.Plot(label='3D topic plot')
   
    with gr.Tab(label='Topic tags'):
        gr.Markdown('## 🏷️ Topic tags')
        gr.Markdown('Download the generated CSV to determine the tags to be given to each article')
        tags_df = gr.DataFrame(headers=['title', 'topics', 'keywords'], wrap=True)
        tags_out = gr.File(label='Download topic tags CSV', file_types=['.csv'], interactive=False)
    
    with gr.Tab(label='Graph'):
        gr.Markdown('## 🕸️ HTML graph file')
        html_out = gr.File(label='Download HTML', file_types=['.html'])

    with gr.Sidebar(width=400):
        gr.Markdown('## 📁 Data upload')
        gr.Markdown('⚠️ The csv must have the columns "<b>title</b>" and "<b>content</b>"')
        csv_in = gr.File(label='CSV', file_types=['.csv'])
        run_btn = gr.Button('Process', variant='primary')

        gr.Markdown('## ⚙️ Parameters')
        gr.Markdown("⚠️ Chunk size must fit in the model's max context!")
        cluster_min = gr.Number(value=2, label='Minimum topic cluster size', interactive=True, minimum=2)
        top_n = gr.Number(value=10, label='Keywords to extract by paper', interactive=True, minimum=1)
        chunk_size = gr.Number(value=1024, label='Chunk size', interactive=True, minimum=64)
        
        gr.Markdown('## 🔮 Embedding Model')
        gr.Markdown('⚠️ To use cloud inference, add the right keys to .env')

        provider = gr.Radio(
            label='Inference provider', 
            choices=[
                ('Ollama', 'ollama'),
                ('HuggingFace Inference', 'hf inference'),
                ('Azure OpenAI', 'azure')
            ], 
            value='ollama', 
            interactive=True
        )

        model_name = gr.Dropdown(
            choices=available_models['ollama']['models'], 
            label=available_models['ollama']['name'], 
            interactive=True,
            allow_custom_value=True
        )

        provider.change(fn=update_models, inputs=[provider], outputs=[model_name])

        gr.Markdown('## 🏷️ Cluster Naming')
        gr.Markdown('⚠️ To use LLM generations, add the right keys to .env')

        naming_method = gr.Radio(
            label='Naming mode',
            choices=[
                ('Centroid (No LLM)', 'centroid'),
                ('Azure OpenAI Prompting', 'azure')
            ],
            value='centroid',
            interactive=True
        )

    run_btn.click(
        fn=pipeline, 
        inputs=[csv_in, model_name, provider, cluster_min, top_n, chunk_size, naming_method], 
        outputs=[plot_2d, plot_3d, html_out, tags_df, tags_out]
    )

if __name__ == "__main__":
    demo.launch()