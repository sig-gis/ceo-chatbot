# ceo-chatbot
Project setup helper (LLM chatbot) for CEO

# 🗺️ Navigating the repo

```{bash}
ceo-chatbot/
├── .venv/                    # uv python virtual environment directory (not tracked; created on first `uv sync` or `uv run`)
├── conf/                     # Configuration files for script arguments (YAML, JSON)
│   ├── base/                 # Global/shared configuration files tracked in repo
│   │   ├── prompts.yml       # Prompt configuration
│   │   └── rag_config.yml    # RAG configuration: models, vectorstore, prompts...
│   └── local/                # Personal/local configs (excluded from version control)
│
├── data/                     # Project data directory (not tracked in repo)
│
├── demo/                     # Try ceo-chatbot in a streamlit app
│
├── notebooks/                # Jupyter notebooks for development, prototyping, demos
│
├── src/                      # Source code (modules and utilities imported by scripts) and scripts
│   └── ceo_chatbot
│       ├── __init__.py
│       ├── config.py               # Utilities for loading confs in conf/base/
│       │
│       ├── ingest                  # Data ingestion pipeline: loaders.py, chunking.py, embeddings.py -> index_builder.py
│       │   ├── __init__.py
│       │   ├── chunking.py         # Utilities for splitting documents into chunks
│       │   ├── embeddings.py       # Utilities for defining embedding model
│       │   ├── index_builder.py    # Define data ingestion pipeline
│       │   └── loaders.py          # Utilities for loading dataset
│       │
│       └── rag                     # RAG pipeline: llm.py, retriever.py -> pipeline.py
│           ├── __init__.py
│           ├── llm.py              # Utilities for defining reader model
│           ├── pipeline.py         # Define RAG pipeline
│           └── retriever.py        # Utilities for doc retrieval and similarity search
│
├── scripts/
│   ├── build_index.py        # Data ingestion pipeline runner (offline); chunk, embed, build vector store
│   └── ...             
│
├── tests/                    # Unit and integration tests
│
├── .gitignore                
├── .python-version           # Python version used by uv environment manager
├── pyproject.toml            # Metadata about the project (for uv)
├── uv.lock                   # Locked dependency versions 
└── README.md                 

```


# 🚀 Getting Started

## 1. Clone the repo

SSH: 
```{bash}
git clone git@github.com:sig-gis/ceo-chatbot.git
```

HTTPS:
```{bash}
git clone https://github.com/sig-gis/ceo-chatbot.git
```

## 2. Manage dependencies with `uv`
   
This app uses uv for dependency managment. 
[Read more about uv in the docs.](https://docs.astral.sh/uv/getting-started/) 

### Install `uv`:

macOS/Linux
```{bash}
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation docs for Windows installation instructions](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)


Also see [CONTRIBUTING.md](CONTRIBUTING.md) for more detail on developing with `uv` in this project. 


## 3. Install ceo-chatbot 

Install the ceo-chatbot package locally in editable mode.

From the project root: 

```{bash}
uv pip install -e .
```

## 4. (offline) Build vector DB

For local development, the knowledge corpus must be set up one time.

If using a gated model such as [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m), 

1. Request access to the model on its Hugging Face model page (access is granted instantly)
2. Generate an access token: Profile > Settings > Access Tokens > + Create new token > Token type Read > Create token
3. Run the following `hf` cli command in your terminal and paste your HF access token when prompted

```{bash}
uv run hf auth login

```

Build the vector DB:

```{bash}
uv run scripts/build_index.py
```

Stay tuned! A planned future version will automate building the vector DB.

## 5. Run a demo chat UI

Launch a basic streamlit application to demo `ceo-chatbot` in a chat UI. 

```{bash}
uv run streamlit run demo/chat_app.py
```
