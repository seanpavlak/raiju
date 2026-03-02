# Contributing to Raiju

Contributions are welcome and appreciated.

Maintainers: see [MAINTAINING.md](MAINTAINING.md) for release steps and one-time setup (Dependabot label, PyPI trusted publishing).

## Getting started

1. Fork the repository and clone your fork.
2. Install in development mode with dev dependencies:

   ```shell
   pip install -e ".[dev]"
   ```

3. Install the pre-commit hooks (Ruff will run on commit):

   ```shell
   pre-commit install
   ```

4. Run the test suite:

   ```shell
   pytest tests/ -v
   ```

## Submitting changes

1. Create a branch for your change.
2. Make your edits, keeping the codebase consistent (Ruff for lint/format).
3. Ensure tests pass.
4. Open a pull request with a clear description of the change.

## Code of conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.
