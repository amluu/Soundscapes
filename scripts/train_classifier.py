"""Train the sound / no-sound gate: logistic regression on segment features.

    python scripts/train_classifier.py --features tfidf --train 1-4 --dev 5 --test 6

--features tfidf uses the vectorizer scripts/match.py saves; minilm uses sentence embeddings.
Splits are by chapter. C is picked by dev F1. Also reports leave-one-chapter-out
(loco) scores and average precision (AP), since dev and test have few sounds.

Writes to --out (default data/processed):
  <id>.<features>.gate.csv     every segment's p_sound and has_sound, from a model refit on all labelled chapters
  <id>.<features>.lr_eval.csv  labelled segments with gold label, dev/test and loco p_sound
  <id>.tfidf.lr_terms.csv      the most sound-like and least sound-like terms (tfidf only)
"""
import argparse
import csv
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from soundscapes.methods import to_csr  # noqa: E402
from soundscapes.tfidf import TfidfVectorizer  # noqa: E402


def chapters(spec):
    """'1-4' -> {1, 2, 3, 4}; '5,7' -> {5, 7}"""
    out = set()
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def scores(y, prob):
    """Precision, recall and F1 at p >= 0.5, plus average precision."""
    p, r, f, _ = precision_recall_fscore_support(y, prob >= 0.5, average='binary', zero_division=0)
    return p, r, f, average_precision_score(y, prob)


def fit(X, y, C):
    return LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(X, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', choices=['tfidf', 'minilm'], default='tfidf')
    ap.add_argument('--id', default='hobbit')
    ap.add_argument('--out', default='data/processed')
    ap.add_argument('--vectorizer', default='data/processed/tfidf.vectorizer.json')
    ap.add_argument('--model', default='models/all-MiniLM-L6-v2', help='MiniLM model folder')
    ap.add_argument('--train', default='1-4')
    ap.add_argument('--dev', default='5')
    ap.add_argument('--test', default='6')
    a = ap.parse_args()

    stem = os.path.join(a.out, a.id)
    with open(f'{stem}.segments.csv', newline='', encoding='utf-8') as f:
        segs = list(csv.DictReader(f))
    with open(f'{stem}.labels.csv', newline='', encoding='utf-8') as f:
        gold = {int(r['id']): int(r['sound']) for r in csv.DictReader(f)}
    texts = [s['text'] for s in segs]
    if a.features == 'tfidf':
        vec = TfidfVectorizer.load(a.vectorizer)
        X = to_csr(vec.transform(texts), len(vec.vocab))
    else:
        from soundscapes.minilm import MiniLM
        X = MiniLM(a.model).encode(texts)
    chap = np.array([int(s['chapter']) for s in segs])
    y = np.array([gold.get(i, -1) for i in range(len(segs))])
    labelled = y >= 0
    split = {name: np.isin(chap, list(chapters(spec))) & labelled
             for name, spec in [('train', a.train), ('dev', a.dev), ('test', a.test)]}
    for name, m in split.items():
        print(f'{name:5} ch {getattr(a, name):4} {m.sum():5} segments, {y[m].sum():3} sound')

    tr, dev, test = split['train'], split['dev'], split['test']
    C = max([0.1, 0.3, 1, 3, 10, 30], key=lambda c: scores(y[dev], fit(X[tr], y[tr], c).predict_proba(X[dev])[:, 1])[2])
    model = fit(X[tr], y[tr], C)
    p = model.predict_proba(X)[:, 1]
    loco = np.zeros(len(y))  # each labelled chapter scored by a model trained on the others
    for c in np.unique(chap[labelled]):
        held = labelled & (chap == c)
        loco[held] = fit(X[labelled & ~held], y[labelled & ~held], C).predict_proba(X[held])[:, 1]

    print(f'\n{a.features}, C = {C} (best dev F1)\n{"":30}{"P":>6}{"R":>6}{"F1":>6}{"AP":>6}')
    for name, m, prob in [('dev', dev, p), ('test', test, p), ('loco', labelled, loco)]:
        for label, pr in [('always no sound', np.zeros(m.sum())), ('logistic regression', prob[m])]:
            print(f'{name:5}{label:25}' + ''.join(f'{v:6.2f}' for v in scores(y[m], pr)))
    print('(AP of "always no sound" is just the share of sound segments)')

    with open(f'{stem}.{a.features}.lr_eval.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'chapter', 'split', 'sound', 'p_sound', 'p_sound_loco', 'text'])
        name = np.full(len(y), '', dtype=object)
        for n, m in split.items():
            name[m] = n
        w.writerows([i, chap[i], name[i], y[i], '' if tr[i] else round(p[i], 3), round(loco[i], 3), texts[i]]
                    for i in np.flatnonzero(labelled))

    if a.features == 'tfidf':
        terms = sorted(vec.vocab, key=vec.vocab.get)
        order = np.argsort(model.coef_[0])
        with open(f'{stem}.tfidf.lr_terms.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'sound_term', 'weight', 'no_sound_term', 'weight'])
            for k in range(30):
                hi, lo = order[-1 - k], order[k]
                w.writerow([k + 1, terms[hi], round(model.coef_[0][hi], 3), terms[lo], round(model.coef_[0][lo], 3)])
        print('\nmost sound-like terms:', ', '.join(terms[j] for j in order[::-1][:15]))

    p = fit(X[labelled], y[labelled], C).predict_proba(X)[:, 1]
    with open(f'{stem}.{a.features}.gate.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'p_sound', 'has_sound'])
        w.writerows([i, round(p[i], 3), int(p[i] >= 0.5)] for i in range(len(segs)))
    print(f'gate: {(p >= 0.5).sum()} of {len(segs)} segments get a sound ({(p >= 0.5).mean():.1%})')


if __name__ == '__main__':
    main()
