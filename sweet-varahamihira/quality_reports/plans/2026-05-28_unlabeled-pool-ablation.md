# Unlabeled-Pool Ablation

Goal: add Carlos-requested ablation over the number of unlabeled volumes.

Plan:
1. Add a compact Results paragraph and table after the existing label-efficiency analysis.
2. Keep $|\mathcal{D}_L|=100$ fixed and vary $|\mathcal{D}_U|\in\{0,100,300,620\}$.
3. Report MRE, SDR$_{2.0}$, SDR$_{4.0}$, and $\Delta$MRE relative to $|\mathcal{D}_U|=0$.
4. Phrase the result cautiously: monotonic improvement with the largest gain at the full pool, not a claim of linear scaling.
5. Synchronize the Chinese review mirror and compile the paper.

