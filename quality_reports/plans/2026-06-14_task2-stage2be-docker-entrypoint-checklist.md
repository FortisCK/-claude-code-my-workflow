# Task2 Stage2BE Docker/Entrypoint Checklist Plan

Date: 2026-06-14

## Goal

Document the current Task2 Docker/submission entrypoint contract so the package
can be converted to an official challenge container quickly once the platform
input/output schema is released.

## Scope

- Current inference command.
- Expected input/output paths.
- Required scripts and weights.
- Preflight and checksum commands.
- Smoke commands.
- Known blockers before final Docker submission.

## Acceptance Criteria

- A checklist exists under `quality_reports/reports/`.
- It clearly separates completed package hardening from official-format
  blockers.
- It gives concrete commands that can be pasted into a Docker smoke test.

