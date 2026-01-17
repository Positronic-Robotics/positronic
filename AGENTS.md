# Contributor guide

## Project context
Positronic is a Python-native toolkit for ML-driven robotics covering the full lifecycle: hardware bring-up, dataset capture, policy training (LeRobot, π₀.₅, GR00T), and inference deployment. Key modules:
- `pimm/` — immediate-mode runtime for control loops and multiprocess orchestration
- `positronic/dataset/` — multi-rate episode recording, lazy transforms, streaming
- `positronic/drivers/` — cameras, grippers, robot arms, WebXR teleoperation
- `positronic/policy/` — policy loading and inference wrappers
- `positronic/training/` — LeRobot conversion and training pipelines

## Contributor behavior
- Don't restore code that you wrote and I deleted
- Don't make commits or git changes unless explicitly asked
- Don't hide errors with try-catch blocks; let failures surface until asked otherwise
- Don't over-engineer: no extra abstractions, features, or "improvements" beyond what's requested
- Don't add comments, docstrings, or type annotations to code you didn't change
- Ask clarifying questions when requirements are ambiguous instead of guessing

## ⚠️ CRITICAL: Always use `uv run` for Python commands

**NEVER run Python directly** (`python`, `python3`, `python -m`). **ALWAYS use `uv run python`** instead.

❌ **WRONG:** `python script.py`, `python3 -m pytest`, `python -c "import foo"`
✅ **CORRECT:** `uv run python script.py`, `uv run pytest`, `uv run python -c "import foo"`

This applies to:
- Running Python scripts
- Running tests with pytest
- Syntax checks with `python -m py_compile`
- Importing modules
- Any Python execution whatsoever

**Why:** This project uses uv for dependency management. Running Python directly bypasses the virtual environment and will cause import errors or use wrong package versions.

## Commands
- Run tests: `uv run pytest --no-cov`
- Run single test file: `uv run pytest path/to/test_file.py --no-cov`
- Lint: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Run any Python: `uv run python script.py`
- Syntax check: `uv run python -m py_compile file.py`

## Testing
- Don't add new test files unless explicitly asked
- Never import inside test functions; add imports at the top of the file

## Commit messages
- Short, imperative sentences (e.g., "Fix wrong type", not "Fixed wrong type")
- Use backticks for code references (e.g., "Fix `RemoteDataset` connection leak")
- No trailing period for short messages
- No Claude/AI attribution

## Available skills
- `/remote-training` — Manage Nebius VMs, Docker images, training jobs, inference servers
- `/push-pr` — Push branch to origin and create PR to upstream with proper workflow

## Related repositories
- `../gr00t` — GR00T model configs and training (Positronic-Robotics/gr00t)
- `../openpi` — OpenPI model integration
- `../internal` — Internal scripts and infrastructure
