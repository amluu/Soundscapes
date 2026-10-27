"""eval graded eval sheets by method, with Cohen's kappa between graders.

    python pipeline/manual-eval/score_eval.py data/processed/hobbit.eval.graded.*.csv
"""
import argparse
import csv
import os
from collections import Counter, defaultdict
from itertools import combinations

from sklearn.metrics import cohen_kappa_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('graded', nargs='+')
    ap.add_argument('--id', default='hobbit')
    ap.add_argument('--out', default='data/processed')
    a = ap.parse_args()

    with open(os.path.join(a.out, f'{a.id}.eval_key.csv'), newline='', encoding='utf-8') as f:
        key = {r['item']: r['methods'] for r in csv.DictReader(f)}
    grades = defaultdict(Counter)
    by_file = {}  # file -> item -> grade
    for path in a.graded:
        with open(path, newline='', encoding='utf-8') as f:
            by_file[path] = {r['item']: r['grade'].strip() for r in csv.DictReader(f) if r['grade'].strip()}
        for item, g in by_file[path].items():
            for m in ([key[item]] if key[item] == 'gate rejected' else key[item].split()):
                grades[m][g] += 1

    print(f'{"":15}{"n":>5}{"1 wrong":>9}{"2 clip":>8}{"3 right":>9}')
    for m, c in sorted(grades.items()):
        n = sum(c.values())
        print(f'{m:15}{n:5}' + ''.join(f'{c[g] / n:{w}.0%}' for g, w in [('1', 9), ('2', 8), ('3', 9)]))

    for (p, x), (q, y) in combinations(by_file.items(), 2):
        both = sorted(x.keys() & y.keys(), key=int)
        if len(both) > 1:
            kappa = cohen_kappa_score([x[i] for i in both], [y[i] for i in both])
            agree = sum(x[i] == y[i] for i in both) / len(both)
            print(f'{os.path.basename(p)} vs {os.path.basename(q)}: {len(both)} items, '
                  f'{agree:.0%} agree, kappa {kappa:.2f}')


if __name__ == '__main__':
    main()
