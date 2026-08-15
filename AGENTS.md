# COCCL Migration Workflow

## Task completion

- Work on the `valid` branch.
- A migration task is complete only after its required verification and report pass.
- Commit only files that belong to the completed task. Do not include unrelated working-tree changes, build products, caches, raw benchmark logs, or mode-only noise.
- After creating the task-end commit, push `valid` to `origin/valid` automatically.
- Never force-push `valid`. If a normal push fails, keep the local commit and report the failure and its cause.
