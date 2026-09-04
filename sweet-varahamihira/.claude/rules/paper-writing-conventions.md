---
paths:
  - "paper/**/*.tex"
---

# Paper Writing Conventions (IEEE TMI)

## Document Class

- Use `ieeecolor` class: `\documentclass[journal,twoside,web]{ieeecolor}` with `\usepackage{generic}`
- Compile with `pdflatex` (NOT xelatex)
- Bibliography style: `IEEEtran.bst` with `\bibliographystyle{IEEEtran}`
- Protected files: `ieeecolor.cls`, `generic.sty` (do not edit)

## Structure

Standard IEEE TMI paper structure:
1. Abstract (no section number)
2. Index Terms
3. Introduction (`\section{Introduction}`, use `\IEEEPARstart{T}{he}` for drop cap)
4. Related Work
5. Method (or "Proposed Method")
6. Experiments (or "Experimental Setup and Results")
7. Discussion
8. Conclusion
9. References

## Citations

- Use `\cite{key}` for numeric references (IEEE style: [1], [2])
- Multiple citations: `\cite{key1, key2, key3}` (renders as [1]-[3] if consecutive)
- Never use `\citet` or `\citep` (those are natbib, not IEEEtran)
- Bibliography keys follow `AuthorYear_keyword` convention

## Notation Discipline

- Define every symbol on first use
- Use `\newcommand` for repeated notation in preamble
- Refer to the notation registry in `knowledge-base-template.md`
- Equations: use `\begin{equation}` for referenced equations, `\[...\]` for unreferenced

## Abbreviation Management

- Define every abbreviation on first use: "transcatheter aortic valve implantation (TAVI)"
- After definition, use abbreviation only
- In the abstract: define abbreviations independently (abstract is standalone)
- Maintain consistency with the terminology table in `knowledge-base-template.md`

## Figure & Table Conventions

- Figures: `\begin{figure}[t]` or `\begin{figure*}[t]` for double-column
- Tables: `\begin{table}[t]` with `\caption` above the table
- Captions must be self-contained and informative
- All figures/tables referenced in text before they appear
- Use `\label{fig:name}` / `\label{tab:name}` and `\ref{fig:name}` / `\ref{tab:name}`

## Cross-References

- Label every section, figure, table, and equation
- Use `\ref{}` (not hardcoded numbers)
- Capitalize "Fig." in running text, "Table" in running text, "Section" before number
