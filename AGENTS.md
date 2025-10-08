# Contributor guide

- Don't restore any code that you wrote and I deleted.
- Don't make any commits or other git changes unless explicitly asked for it.
- Don't try to hinder and hide errors or exceptions. If something fails, it is better to fail publicly, until I explicitly ask you to take care of extra errors.

## Local environment
- Use `uv` tool to manage python environment

## Testing insturctions
- Find the CI plan in .github/workflows directory.
- The work is done in virtual environment (`.venv/bin/activate`). You may need to initialise it when running scripts.
- Run tests locally using `pytest` command. Disable coverage with `--no-cov` argument.
- Add or update tests for the code you change, even if nobody asked.
- Don't add new test files unless explicitly asked
