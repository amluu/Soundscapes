"""Part-of-speech-tagged words: "he saw" -> saw_v, "a hand saw" -> saw_n.

Uses NLTK's perceptron tagger. Penn tags are collapsed to n, v, a (adjective)
and r (adverb); other words are dropped along with stop words.
"""
import re

from .tfidf import STOP_WORDS

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_COARSE = {'N': 'n', 'V': 'v', 'J': 'a', 'R': 'r'}
_tagger = None


def pos_tokenize(text):
    global _tagger
    if _tagger is None:
        from nltk.tag.perceptron import PerceptronTagger
        _tagger = PerceptronTagger()
    words = [w[:-2] if w.lower().endswith("'s") else w for w in _WORD.findall(text)]
    # Tag the full sentence (stop words included) so the tagger has context.
    tagged = _tagger.tag(words) if words else []
    return [f'{w.lower()}_{_COARSE[t[0]]}' for w, t in tagged
            if t[0] in _COARSE and len(w) > 1 and w.lower() not in STOP_WORDS]
