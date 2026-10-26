"""TF-IDF over words and word pairs, in plain Python.

Weights match scikit-learn's TfidfVectorizer: idf = ln((1 + n) / (1 + df)) + 1,
raw counts, L2-normalized rows. Rows are sparse dicts {column: weight}, so the
dot product of two rows is their cosine similarity.
"""
import json
import math
import re
from collections import Counter, defaultdict

STOP_WORDS = frozenset('''
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each else ever every few for from further
had has have having he her here hers herself him himself his how however if in into is it its itself
just let may me might more most much must my myself no nor not now of off on once only or other ought
our ours ourselves out over own perhaps quite rather same shall she should so some still such than that
the their theirs them themselves then there these they this those though through thus to too under until
up upon us very was we well were what when where whether which while who whom whose why will with would
yet you your yours yourself yourselves
'''.split())

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text):
    """Lowercase words minus stop words; possessive 's dropped."""
    words = [w[:-2] if w.endswith("'s") else w for w in _WORD.findall(text.lower())]
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


def with_pairs(words):
    """Words plus adjacent word pairs. Stop words are already gone, so "crack of thunder" -> "crack thunder"."""
    return words + [f'{a} {b}' for a, b in zip(words, words[1:])]


class TfidfVectorizer:
    def __init__(self, min_df=2, pos=False):
        self.min_df, self.pos = min_df, pos
        self.vocab, self.idf = {}, []

    def terms(self, text):
        if self.pos:
            from .postag import pos_tokenize
            return with_pairs(pos_tokenize(text))
        return with_pairs(tokenize(text))

    def fit(self, docs):
        df = Counter(t for d in docs for t in set(self.terms(d)))
        kept = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab = {t: i for i, t in enumerate(kept)}
        self.idf = [math.log((1 + len(docs)) / (1 + df[t])) + 1 for t in kept]
        return self

    def transform(self, docs):
        return [self._row(d) for d in docs]

    def _row(self, text):
        tf = Counter(self.vocab[t] for t in self.terms(text) if t in self.vocab)
        row = {j: c * self.idf[j] for j, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in row.values()))
        return {j: w / norm for j, w in row.items()} if norm else {}

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'min_df': self.min_df, 'pos': self.pos, 'terms': sorted(self.vocab, key=self.vocab.get),
                       'idf': self.idf}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        v = cls(d['min_df'], d['pos'])
        v.vocab = {t: i for i, t in enumerate(d['terms'])}
        v.idf = d['idf']
        return v


def sparse_top_k(queries, corpus, k):
    """Top-k cosine matches per query row, using an inverted index over the corpus."""
    index = defaultdict(list)
    for i, row in enumerate(corpus):
        for j, w in row.items():
            index[j].append((i, w))
    out = []
    for q in queries:
        scores = defaultdict(float)
        for j, w in q.items():
            for i, w2 in index.get(j, ()):
                scores[i] += w * w2
        out.append(sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k])
    return out
