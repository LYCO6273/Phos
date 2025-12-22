# Repository Guidelines

## Project Structure

- `Phos_0.1.1 copy.py`: primary Streamlit app entrypoint (used by the devcontainer).
- `Phos_0.1.1.py`: alternate/older variant of the app.
- `legacy/`: archived versions (e.g., `legacy/Phos_0.1.0.py`) kept for reference.
- `.streamlit/config.toml`: Streamlit server limits (upload/message size).
- `.devcontainer/devcontainer.json`: Codespaces/devcontainer setup and default run command.

## Build, Test, and Development Commands

- Install deps: `python -m pip install -r requirements.txt`
- Run locally: `streamlit run "Phos_0.1.1 copy.py"`
- Devcontainer run (matches config): `streamlit run "Phos_0.1.1 copy.py" --server.enableCORS false --server.enableXsrfProtection false`
- Quick sanity check (no test suite yet): `python -m py_compile "Phos_0.1.1 copy.py"`

Notes: README states Python `3.13`, while `.devcontainer` uses Python `3.11`. If you update dependencies or language features, keep both in mind and document the expected version in `README.md`.

## Coding Style & Naming Conventions

- Python: 4-space indentation, keep functions in `snake_case`, constants in `UPPER_SNAKE_CASE` when introduced.
- Prefer small, pure helper functions for image ops; avoid duplicating parameter tables (e.g., film presets) across files.
- Keep UI strings and slider defaults near the Streamlit layout code to make UX edits easy.

## Testing Guidelines

- There is no automated test framework configured. If you add tests, use `pytest` and place them under `tests/` (e.g., `tests/test_tonemapping.py`).
- Focus on deterministic units (tone mapping, grain generation) and avoid depending on Streamlit runtime in tests.

## Commit & Pull Request Guidelines

- Use short, imperative commit subjects (examples in history: “Update requirements.txt”, “Added Dev Container Folder”).
- Avoid accidental commit messages like “Changes to be committed…”.
- PRs should include: a concise summary, how to run (`streamlit run ...`), and screenshots/GIFs for UI or rendering changes. Link related issues when applicable.

## Security & Configuration Tips

- Do not disable CORS/XSRF outside local development. If you change `.streamlit/config.toml`, explain the impact on uploads and memory usage.
- Never commit API keys, credentials, or private images used for debugging.
