"""The four matchers. Each embeds segments and captions into one shared space
and returns (segment_vectors, caption_vectors, extra_files_to_save).

    tfidf      words + word pairs, fitted on segments and captions together
    tfidf_pos  the same, with part-of-speech-tagged words
    lsa        TF-IDF reduced to 300 dimensions with truncated SVD
    minilm     all-MiniLM-L6-v2 sentence embeddings
"""
import numpy as np

from .tfidf import TfidfVectorizer, sparse_top_k


def tfidf(segs, caps, pos=False, **_):
    vec = TfidfVectorizer(pos=pos).fit(segs + caps)
    return vec.transform(segs), vec.transform(caps), {'vectorizer': vec}


def tfidf_pos(segs, caps, **_):
    return tfidf(segs, caps, pos=True)


def to_csr(rows, n_cols):
    """Sparse dict rows {column: weight} -> scipy CSR matrix"""
    from scipy.sparse import csr_matrix
    return csr_matrix(([w for r in rows for w in r.values()], ([i for i, r in enumerate(rows) for _ in r],
                                                               [j for r in rows for j in r])),
                      shape=(len(rows), n_cols))


def lsa(segs, caps, dims=300, **_):
    from sklearn.decomposition import TruncatedSVD
    from .minilm import normalize
    S, C, extra = tfidf(segs, caps)
    X = to_csr(S + C, len(extra['vectorizer'].vocab))
    Z = normalize(TruncatedSVD(dims, random_state=0).fit_transform(X).astype(np.float32))
    return Z[:len(S)], Z[len(S):], {}


def minilm(segs, caps, model_dir='models/all-MiniLM-L6-v2', **_):
    from .minilm import MiniLM
    m = MiniLM(model_dir)
    return m.encode(segs), m.encode(caps), {}


METHODS = {'tfidf': tfidf, 'tfidf_pos': tfidf_pos, 'lsa': lsa, 'minilm': minilm}


def top_k(S, C, k=3):
    """For each segment, its k best captions as [(caption_index, cosine), ...], best first."""
    if isinstance(S, list):
        return sparse_top_k(S, C, k)
    out = []
    for b in range(0, len(S), 1024):
        sim = S[b:b + 1024] @ C.T
        best = np.argsort(-sim, axis=1)[:, :k]
        out += [[(int(i), float(r[i])) for i in row if r[i] > 0] for row, r in zip(best, sim)]
    return out


def cutoff_for_share(matches, share):
    """The score cutoff at which `share` of all segments keep a sound."""
    best = sorted((m[0][1] for m in matches if m), reverse=True)
    n = round(share * len(matches))
    return best[n - 1] if 0 < n <= len(best) else float('inf')
