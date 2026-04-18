# MASS-AI

Web-first risk intelligence for smart meter anomaly detection and pilot analytics.

## Branch Guide

- `main` is the web-facing branch.
- `desktop-local` preserves the old local desktop version.
- If you need the old desktop app, switch branches with `git checkout desktop-local`.

## What Lives Here

- Root landing: [index.html](/C:/Users/kocak/.codex/worktrees/1b2e/MASS_AI_UNIFIED_APP_CLEAN/index.html)
- Web app: [project/web/dashboard/app.py](/C:/Users/kocak/.codex/worktrees/1b2e/MASS_AI_UNIFIED_APP_CLEAN/project/web/dashboard/app.py)
- Shared engine: [project/core](/C:/Users/kocak/.codex/worktrees/1b2e/MASS_AI_UNIFIED_APP_CLEAN/project/core)
- Tests: [project/tests](/C:/Users/kocak/.codex/worktrees/1b2e/MASS_AI_UNIFIED_APP_CLEAN/project/tests)
- Asset intake for approved web visuals: [images/README.md](/C:/Users/kocak/.codex/worktrees/1b2e/MASS_AI_UNIFIED_APP_CLEAN/images/README.md)

## Quick Start

```powershell
.\INSTALL_REQUIREMENTS.bat
.\START_MASS_AI.bat
```

Direct web launch:

```powershell
.\START_MASS_AI_WEB.bat
```

Smoke checks:

```powershell
.\RUN_SMOKE_TESTS.bat
```

## Repo Shape

- `project/web/` contains the current Streamlit app and web-specific dependencies.
- `project/core/` keeps shared domain, engine, and persistence helpers.
- `project/data/` stores demo and processed datasets used by the web flow.
- `project/archive/` keeps older research code that is no longer part of the main runtime path.
- `docs/` holds product and rollout notes for the web-facing pilot story.

## Visual Assets

Old desktop screenshots were removed from `main` on purpose. This branch is ready for approved web visuals only.

When new mockups are available, drop in:

- `1` hero-quality web mockup
- `3-6` supporting web screens
- optional captions or ordering notes

The expected intake location is documented in [images/README.md](/C:/Users/kocak/.codex/worktrees/1b2e/MASS_AI_UNIFIED_APP_CLEAN/images/README.md).
