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

## Commands
- Run tests: `uv run pytest --no-cov`
- Run single test file: `uv run pytest path/to/test_file.py --no-cov`
- Lint: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Run any Python: `uv run python script.py`

## Testing
- Don't add new test files unless explicitly asked
- Never import inside test functions; add imports at the top of the file
