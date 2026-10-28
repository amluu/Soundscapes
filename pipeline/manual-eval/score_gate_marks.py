"""Tally the group's real-sound marks on the gate picks, with Cohen's kappa.

    python pipeline/manual-eval/score_gate_marks.py hobbit.gate_marks.*.csv
"""
import csv
import sys
from collections import Counter
from itertools import combinations

from sklearn.metrics import cohen_kappa_score

marks = {}  # grader -> segment id -> row
for path in sys.argv[1:]:
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            marks.setdefault(r['grader'], {})[r['id']] = r

real, n = Counter(), Counter()
for rows in marks.values():
    for r in rows.values():
        part = 'ch 1-6 extra' if int(r['chapter']) <= 6 else 'ch 7-19'
        for m in ['tfidf', 'minilm']:
            if r[m] == '1':
                n[m, part] += 1
                real[m, part] += int(r['real_sound'])
for (m, part), k in sorted(n.items()):
    print(f'{m:7} {part:13} {real[m, part]:4} of {k:4} marked picks are real sounds ({real[m, part] / k:.0%})')

for (a, x), (b, y) in combinations(sorted(marks.items()), 2):
    both = sorted(x.keys() & y.keys(), key=int)
    if len(both) > 1:
        xs, ys = [x[i]['real_sound'] for i in both], [y[i]['real_sound'] for i in both]
        agree = sum(p == q for p, q in zip(xs, ys)) / len(both)
        print(f'{a} vs {b}: {len(both)} rows, {agree:.0%} agree, kappa {cohen_kappa_score(xs, ys):.2f}')
