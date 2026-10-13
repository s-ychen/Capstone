# Capstone
GoalPath Bond

## Setup

Install [uv](https://docs.astral.sh/uv/) once per machine:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh                                     # macOS/Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

Then sync the environment from `uv.lock`:

```sh
uv sync
```

## Usage

```sh
uv run python library/strategies.py   # run without activating
source .venv/bin/activate             # or activate the env (Windows: .venv\Scripts\activate)
uv add <package>                      # add a dependency, commit pyproject.toml + uv.lock
```

Run `uv sync` again after pulling changes to `uv.lock`.
