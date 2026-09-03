# Update Chinese Mirror

Date: 2026-05-16

Goal: Update `paper/main_zh.md` so that it matches the current scientific content of `paper/main.tex`.

Planned edits:

1. [x] Rebuild the Chinese mirror around the current paper structure and corrected final numbers.
2. [x] Remove stale claims from the old Chinese file, including `2.28`, `18.3%`, five downstream measurements, old RCO-underpowered wording, old P1-regression limitation, old uncertainty calibration/scatter story, and standalone fusion-result framing.
3. [x] Keep author/affiliation/acknowledgment placeholders explicitly marked as pending, because those require user/advisor decisions.
4. [x] Verify by grepping for known stale terms and checking the markdown section structure.

Notes:

- `paper/main.tex` remains the single source of truth.
- The Chinese file is for reading/review support, not a compiled submission artifact.

## Result

Implemented in `paper/main_zh.md`.

- Rebuilt the file as a 354-line current-version Chinese reading mirror.
- Updated the abstract, contributions, method summary, all major result tables, clinical downstream section, discussion, limitations, and conclusion to match the current English manuscript.
- Preserved pending author/affiliation/acknowledgment/biography status explicitly.
- Kept fusion only as the final-output definition and removed the standalone Student1/Student2/Fuse result framing.

Verification:

- Grepped for known stale claims: no `2.28` result claim, `18.3%`, `22.6`, `2.16`, five-measurement wording, old RCO-underpowered wording, stale uncertainty-error calibration story, or `tab:fusion` remains.
- Confirmed current final numbers are present: `2.19`, `21.5%`, six downstream measurements, `46.7%` MSL discordance with `0% critical`, LCO sensitivity `75.0%`, RCO `n+=1`, annular-area `9.4%` under-prediction.
- Checked markdown section structure from Abstract through author information.
