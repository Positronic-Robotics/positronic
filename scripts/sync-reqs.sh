uv pip compile pyproject.toml -o requirements.txt
uv pip compile pyproject.toml -o requirements-hardware.txt --extra hardware --extra lerobot
uv pip compile pyproject.toml -o requirements-all.txt --all-extras
