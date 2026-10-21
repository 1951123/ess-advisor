# Agent instructions — ESS Advisor

Applies only to `1951123/ess-advisor`. This repository independently owns advisor algorithms, admissibility/maintenance-cost components and software tests. Cross-repository research definitions and experiment approvals are coordinated in [ess-research](https://github.com/1951123/ess-research); PostgreSQL source is owned by [ess-postgresql](https://github.com/1951123/ess-postgresql).

## Worktree and environment

- Use one Issue → one branch → one Git worktree → one Codex session → one Draft PR; keep `main` stable. Do not modify another repository by default.
- Optional worktree parent: `ESS_ADVISOR_WORKTREE_PARENT="$HOME/projects/ess-advisor/.worktrees"`. If absent, require a specific path instead of guessing. The older `ESS_WORKTREE_PARENT` belongs to ess-research.
- Each worktree has an independent, ignored `.venv`: `python3 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, `.venv/bin/python -m pip install -e .`. Run targeted checks with that interpreter; record versions and outcomes.
- Shared original data uses `ESS_DATA_ROOT` only when needed; derived run artifacts use an explicitly verified `ESS_RUNTIME_PARENT` child separated by repo / Issue / run. Source `~/.bashrc` and verify needed variables in the **same shell invocation**, since Codex shells may be noninteractive. Do not edit the user's shell profile.

## Scientific and execution boundaries

- Keep the general admissibility, estimation, maintenance-cost and optimization components separate. Preserve clear distinction between exact Fraction reference results and numerical MILP status; never label an unproven incumbent globally optimal.
- Reuse existing small reference instances and focused tests first. Do not add frameworks, new benchmarks, persistent ledgers or redundant gates without a specific correctness need.
- When consuming another repository, identify its immutable full Git commit SHA and relevant artifact/interface version. Do not track moving branch HEAD as the scientific dependency; do not copy another repo's code here.
- Researcher authorization is required for full calibration, expensive PG operations, or official held-out/test evaluation. Agent review and ordinary unit-test permission do not imply experiment approval. Never use held-out outcomes to choose resources.
- Do not commit `.venv`, raw datasets, generated large experiment products, PostgreSQL binaries, fixed ANALYZE samples, logs or secrets. Preserve compact reproducibility evidence and explicitly report failures/limits.
- Keep each PR Draft until researcher review; no autonomous merge or destructive worktree cleanup.
