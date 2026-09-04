# Plan: Oyama-Era Doraemon Codex Pet

Date: 2026-05-16

Goal: Recreate the pet as a compact Doraemon companion with an Oyama-era anime feel: rounder proportions, softer vintage TV linework, flatter color, cream-white face and belly, slightly goofy expression, red nose, collar bell, and simple pocket.

Checklist:

1. Reset previous pet attempt.
   - Remove the rejected sticker-style run artifacts and generated images.
2. Prepare Oyama-era Doraemon pet run.
   - Create a new hatch-pet run with style notes based on the user's reference image.
3. Generate Oyama-era base look.
   - Generate a centered full-body base image on a removable chroma-key background.
   - Save it as the canonical reference.
4. Generate animation rows.
   - Generate all required pet state strips while preserving the base identity.
   - Mirror running-left from running-right only if visually safe.
5. Validate and package pet.
   - Extract frames, inspect components, compose the atlas, validate, create QA previews, and package `pet.json` plus `spritesheet.webp`.

Verification:

- Run hatch-pet validation scripts.
- Visually inspect the contact sheet and previews for consistent vintage Doraemon identity.
