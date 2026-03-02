# Contributing to Positronic

Thank you for your interest in contributing to Positronic! We welcome contributions from the community.

## License Grant

By submitting a pull request, you agree that your contributions will be licensed under the Apache License 2.0 (the same license as this project). You confirm that you have the right to grant this license and that your contribution does not violate any third-party rights.

## Commit Signing (SSH/GPG)

We **require** all commits to be cryptographically signed to verify the author's identity.

### Setting up SSH Signing (Recommended)

1. Generate an SSH key (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```

2. Configure Git to use SSH signing:
   ```bash
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/id_ed25519.pub
   git config --global commit.gpgsign true
   ```

3. Add your SSH key to GitHub as a signing key (Settings → SSH and GPG keys → New SSH key → select "Signing Key")

### Setting up GPG Signing (Alternative)

1. Generate a GPG key (if you don't have one):
   ```bash
   gpg --full-generate-key
   ```

2. List your keys and copy the key ID:
   ```bash
   gpg --list-secret-keys --keyid-format=long
   ```

3. Configure Git to use your key:
   ```bash
   git config --global user.signingkey YOUR_KEY_ID
   git config --global commit.gpgsign true
   ```

4. Add your GPG key to GitHub (Settings → SSH and GPG keys → New GPG key)

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality and enforce policies.

### Installation

```bash
uv pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type post-commit
```

This will automatically run checks before each commit, including:
- **Ruff linting and formatting** - Code style and quality checks
- **Commit signature verification** - Ensures commits are cryptographically signed
- **YAML/TOML validation** - Checks configuration files
- **Large file detection** - Prevents accidentally committing large files

### Running Manually

You can run pre-commit checks manually on all files:

```bash
pre-commit run --all-files
```

## Development Workflow

1. Fork the repository and clone your fork
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Install development dependencies:
   ```bash
   uv sync --extra dev
   ```

4. Install pre-commit hooks:
   ```bash
   uv pip install pre-commit
   pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type post-commit
   ```

5. Make your changes and ensure they pass all checks:
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run pytest
   ```

6. Commit your changes with signature:
   ```bash
   git commit -S -m "Add your feature"
   ```

7. Push to your fork and create a pull request

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

- **Line length**: 120 characters
- **Quote style**: Single quotes
- **Target Python version**: 3.11+

Run these commands before committing:

```bash
uv run ruff check --fix .   # Auto-fix linting issues
uv run ruff format .         # Format code
```

## Testing

All new features should include tests. Run the test suite with:

```bash
uv run pytest
```

With coverage report:

```bash
uv run pytest --cov=positronic --cov=pimm --cov-report=term-missing
```

## Pull Request Guidelines

- Ensure all commits are cryptographically signed
- Keep PRs focused on a single feature or bugfix
- Include tests for new functionality
- Update documentation as needed
- All CI checks must pass
- Follow the existing code style

## Questions?

If you have questions or need help, please open an issue on GitHub.
