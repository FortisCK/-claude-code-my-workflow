# Post-Retrain Paper Update — TODO

**Last updated:** 2026-05-15

---

## Phase 1 — Number updates

- [x] All TRUST numbers in tables + Abstract + §4.3 inline text

## Phase 2 — Narrative + figures (can do without new data)

- [x] §4.5 per-landmark narrative + §4.4 calibration + §5 limitations + Fig 4 caption (P1 story)
- [x] **Phase 2.2** — Abstract closing: remove/rewrite "underpowered on right ostium"
- [ ] **Phase 2.3** — §5 limitations: remove/soften the "RCO underpowered" item
- [ ] **Phase 2.4** — Regenerate Figure 4 (per-landmark grouped bar) from new MRE data
- [ ] **Phase 2.5** — Figure 5 (qualitative): select new representative cases + regenerate

## Phase 3 — Needs new data from server / Leo

- [ ] §4.4 downstream tables (MSL / coronary height / calcification / C-arm / DLZ) — waiting for new clinical measurements
- [ ] Table VII r/ū + Figure 3 uncertainty scatter — waiting for new per-landmark uncertainty
- [ ] Confirm Stage 1 numbers (6.74 mm, 298/300) — retrained or kept?
- [ ] Confirm Label efficiency Table VI |D_L|=20, 50 rows — retrained or kept?
- [ ] Annulus polygons — write .pf parser + add 6th downstream measurement

## Housekeeping

- [ ] Clean up `bib_entries_to_add.bib` (26 entries — merge or delete?)
- [ ] Replace placeholder author block + affiliations + biographies
- [ ] Re-measure inference time (~250 ms estimate) when server is free

---

**Next up:** Housekeeping / remaining external confirmations (author block, bibliography cleanup, Stage 1 and label-efficiency confirmation, inference-time re-measurement)
