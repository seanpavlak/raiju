# Maintainer guide

One-time setup and release steps for maintainers.

## One-time setup

### 1. Create the `dependencies` label

So Dependabot PRs are easy to filter:

1. Open **Issues** (or **Pull requests**) → **Labels** → **New label**.
2. Name: `dependencies`, choose a color, save.

### 2. Configure PyPI trusted publishing

So the [pypi-publish](.github/workflows/pypi-publish.yml) workflow can publish without storing a password:

1. On [PyPI](https://pypi.org), open your project **raiju** (or create it).
2. Go to **Publishing** → **Add a new trusted publisher**.
3. **PyPI Project name:** `raiju`.
4. **Owner:** `seanpavlak`.
5. **Repository name:** `raiju`.
6. **Workflow name:** `pypi-publish.yml`.
7. **Environment name:** leave empty (or create a `pypi` environment if you prefer).
8. Add the publisher.

After this, pushing a version tag (e.g. `v0.1.0`) will run the workflow and publish to PyPI.

## Releasing

1. Update [CHANGELOG.md](CHANGELOG.md): move items from `[Unreleased]` into a new `[X.Y.Z]` section and set the date.
2. Bump version in [pyproject.toml](pyproject.toml) to match.
3. Commit, push, then create and push the tag:

   ```shell
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. The **Release** workflow will create the GitHub Release; the **pypi-publish** workflow will publish to PyPI (after trusted publishing is configured).
