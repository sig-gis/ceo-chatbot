# Developing with `uv`

### Run scripts with `uv`

```{bash}
uv run python3 <script-name>.py
```

This approach does the following: 
- Create .venv if needed
- Sync dependencies
- Run script

#### Optional: work in a manually activated environment 

If you prefer to work in a manually activated environment:

```{bash}
uv sync
source .venv/bin/activate
python3 <script-name>.py
```

#### Optional: work in a jupyter notebook

*NOTE: the required ipykernel and uv dependencies to follow the steps below are already loaded in this project via `uv add --dev ipykernel uv`*

- Create a new jupyter notebook inside `ceo-chatbot/notebooks/`
- When prompted to select a kernel, choose "Python Environments" and select the project's virtual environment

Further, you can modify the project's environment experimentally in notebooks without updating the project's `pyproject.toml` or `uv.lock` files. 

Within a notebook running the project's python `.venv`, you can run uv commands with shell escapes and magics, e.g. `!uv pip install <package>` or `%pip install <package>`.

*Note*: running `!uv add <package>` in this scenario *will* add <package> to the project's dependencies and modify the project's `pyproject.toml` and `uv.lock` files.  

[See the docs for more details](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code) on how to use jupyter from VS Code in a uv-managed project. 

### Modifying the `uv` environment

#### Sync dependencies when pulling new changes

When you pull changes that modify `pyproject.toml` or `uv.lock`, re-sync your local environment:

```{bash}
uv sync
```

#### Add/update dependencies (and track them in version control)

If your code modifications require updates to the dependencies... (see [Managing Dependencies in uv docs](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)), e.g.

- `uv add <package>` # add a dependency
- `uv lock --upgrade-package <package>` # upgrade a dependency
- `uv remove <package>` # remove a dependency
- `uv add --dev <package>` # add a dev-only package (e.g. ipykernel, which is useful for dev notebooks but not in deployment)

... You will need to commit:
- `pyproject.toml` 
- `uv.lock` 

Note: `.venv` is ignored; it is generated locally based on `pyproject.toml` and `uv.lock`