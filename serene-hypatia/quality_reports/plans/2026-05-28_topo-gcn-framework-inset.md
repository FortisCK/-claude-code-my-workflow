# Topo-GCN Framework Inset

Goal: create a code-generated Topo-GCN inset that can replace the generic GCN icon in Fig. 2.

Rationale:
- The current framework icon shows a generic graph, but the method defines a fixed 10-node anatomical graph.
- The inset should show the actual landmark topology while remaining readable at framework scale.

Plan:
1. Use the paper's graph definition as the source of truth.
2. Draw the annular ring as the alternating hinge/commissure cycle:
   `P0-P3-P1-P4-P2-P5-P0`.
3. Draw `P6` at the center and connect it to all landmarks.
4. Draw coronary ostia as cusp-adjacent external nodes:
   `P8` connected to `P2/P4/P5`, and `P9` connected to `P1/P3/P4`.
5. Draw `P7` near the NCC-RCC region, connected to `P0/P1/P3/P6`.
6. Export vector PDF and high-resolution PNG into `figs/`.

