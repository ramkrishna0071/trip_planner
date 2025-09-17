# Agent Guidelines

Welcome! This repository targets technically inclined readers (data scientists
or early-career developers). Please keep the following in mind when editing
files under this root:

1. **Testing** – Run `uv run pytest` (or `pytest` in your active environment)
   before submitting changes. Include the command and outcome in your summary.
2. **Logging** – Prefer structured, human-readable log lines. Reuse the module
   loggers that respect the `TRIP_PLANNER_LOG_LEVEL` environment variable.
3. **Comments** – Add short explanatory comments when introducing new control
   flow so that readers can follow the orchestration pipeline.
4. **Dependencies** – Update `pyproject.toml` and `uv.lock` together if you add
   new packages.
5. **Documentation** – Keep `README.md` current whenever behaviour or setup
   steps change.

Thanks for helping keep the planner transparent and easy to operate!
