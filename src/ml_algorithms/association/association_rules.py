"""
Apriori and FP-Growth Association Rule Mining from Scratch
"""
from collections import defaultdict

class AprioriFromScratch:
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fit(self, transactions):
        n_trans = len(transactions)
        item_counts = defaultdict(int)

        for trans in transactions:
            for item in set(trans):
                item_counts[frozenset([item])] += 1

        freq_itemsets = {}
        current_l = {item: count / n_trans for item, count in item_counts.items() if (count / n_trans) >= self.min_support}
        freq_itemsets.update(current_l)

        k = 2
        while current_l:
            candidates = self._generate_candidates(list(current_l.keys()), k)
            c_counts = defaultdict(int)

            for trans in transactions:
                t_set = set(trans)
                for cand in candidates:
                    if cand.issubset(t_set):
                        c_counts[cand] += 1

            current_l = {cand: count / n_trans for cand, count in c_counts.items() if (count / n_trans) >= self.min_support}
            freq_itemsets.update(current_l)
            k += 1

        return freq_itemsets

    def _generate_candidates(self, itemsets, k):
        candidates = set()
        n = len(itemsets)
        for i in range(n):
            for j in range(i + 1, n):
                union = itemsets[i].union(itemsets[j])
                if len(union) == k:
                    candidates.add(union)
        return candidates

# Revision commit 2026-07-09 #1
