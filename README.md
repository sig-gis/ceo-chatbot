# ceo-chatbot
Project setup helper (LLM chatbot) for CEO

# 🗺️ Navigating the repo

```{bash}
ceo-chatbot/
├── .venv/                    # uv python virtual environment directory (not tracked; created on first `uv sync` or `uv run`)
├── conf/                     # Configuration files for script arguments (YAML, JSON)
│   ├── base/                 # Global/shared configuration files tracked in repo
│   └── local/                # Personal/local configs (excluded from version control)
│
├── data/                     # Project data directory (not tracked in repo)
│
├── notebooks/                # Jupyter notebooks for development, prototyping, demos
│
├── scripts/                  # Python scripts for automation, orchestration, and deployment
│
├── src/                      # Source code (modules and utilities imported by scripts)
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

## 2. Install `uv`
   
This app uses uv for dependency managment. 
[Read more about uv in the docs.](https://docs.astral.sh/uv/getting-started/) 

Install `uv`:

macOS/Linux
```{bash}
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation docs for Windows installation instructions](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)


### 2b. (Optional) Manually activate the `uv` environment

You can skip this if you prefer to use `uv run` in Step 3.

If you prefer to work in a manually activated environment:

```{bash}
uv sync
source .venv/bin/activate
```

This creates and activates the .venv, syncing dependencies from pyproject.toml and uv.lock.



## 3. Run scripts

### ✅ Option A (Recommended): Without Manual Activation

This is the simplest method. It will:

- Create .venv if needed
- Sync dependencies
- Run script

```{bash}
uv run python3 <script-name>.py
```

### Option B: With Activated Environment 

If you’ve activated the environment manually (see 2b):

```{bash}
python3 <script-name>.py
```

## 4. Contributing 

### A. Sync dependencies when pulling new changes

When you pull changes that modify `pyproject.toml` or `uv.lock`, re-sync your local environment:

```{bash}
uv sync
```

### B. Add/update dependencies (and track them in version control)

If your code modifications require updates to the dependencies... (see [Managing Dependencies in uv docs](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)), e.g.

- `uv add <package>` # add a dependency
- `uv lock --upgrade-package <package>` # upgrade a dependency
- `uv remove <package>` # remove a dependency
- `uv add --dev <package>` # add a dev-only package (e.g. ipykernel, which is useful for dev notebooks but not in deployment)

... You will need to commit:
- `pyproject.toml` 
- `uv.lock` 

Note: `.venv` is ignored; it is generated locally based on `pyproject.toml` and `uv.lock`

### C. (dev) Connect Notebooks to the uv project's virtual environment

*NOTE: the required ipykernel and uv dependencies to follow the steps below are already loaded in this project via `uv add --dev ipykernel uv`*

- Create a new jupyter notebook inside `ceo-chatbot/notebooks/`
- When prompted to select a kernel, choose "Python Environments" and select the project's virtual environment

Further, you can modify the project's environment experimentally in notebooks without updating the project's `pyproject.toml` or `uv.lock` files. 

Within a notebook running the project's python `.venv`, you can run uv commands with shell escapes and magics, e.g. `!uv pip install <package>` or `%pip install <package>`.

*Note*: running `!uv add <package>` in this scenario *will* add <package> to the project's dependencies and modify the project's `pyproject.toml` and `uv.lock` files.  

[See the docs for more details](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code) on how to use jupyter from VS Code in a uv-managed project. 
