# Plan: replace the multi-reader downstream results

1. Audit the current English and Chinese multi-reader table and all dependent claims.
2. Replace the old P0--P7 reader comparison table with the supplied six-comparison downstream table while leaving the main 30-case test-set description unchanged.
3. Revise the Results, Discussion, Limitations, and Conclusion conservatively so they describe measurement-level multi-reader context without claiming a complete 10-landmark localization analysis.
4. Synchronize the Chinese review copy with the authoritative LaTeX source.
5. Compile the paper, check references and layout warnings, and inspect the rendered pages containing the revised table.

## Corrected-result addendum

The first supplied six-row downstream table was subsequently identified as potentially problematic. Reader A1 is the original ground-truth annotation, so its TRUST comparison is already represented by the primary paper results and should not be duplicated. Replace that provisional table with the corrected three-comparison results:

- Reader B (Jean) vs Reader A2 (Leo repeat annotation).
- TRUST vs Reader B.
- TRUST vs Reader A2.

Report both full 10-landmark localization metrics and downstream MAEs. Retain no numerical claims from the provisional A1/A2/B table unless they appear in the corrected results.

Constraints:

- Preserve the current redline convention for newly revised text.
- Do not claim multi-center validation.
- Do not invent sample sizes, standard deviations, confidence intervals, or point-level P8/P9 results that have not been supplied.
- Keep the official 100/20/30 dataset split and the main test-set results unchanged in this pass.
