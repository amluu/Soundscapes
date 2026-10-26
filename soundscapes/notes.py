"""Read a Kindle notebook export and line its highlights up with our segments.

Each highlight is a line from the book; the note(s) after it describe its sound.
The Kindle edition differs from our text in punctuation and the odd typo,
so both sides are reduced to bare letters and aligned with difflib.
"""
import difflib
import html
import re

MIN_SCORE = 0.8    # share of the highlight's letters that must be found in our text
MIN_OVERLAP = 15   # letters a segment must share with a highlight to count (or half the segment)


def read_notebook(path):
    """[{'chapter', 'highlight', 'notes': [str, ...]}] in reading order.

    A note belongs to the highlight before it. A note can hold two sounds, one per line.
    """
    with open(path, encoding='utf-8') as f:
        doc = f.read()
    items, chapter = [], 0
    for cls, body in re.findall(r'<div class="(sectionHeading|noteHeading|noteText)">(.*?)</div>', doc, re.S):
        body = html.unescape(re.sub(r'<[^>]+>', '', body)).strip()
        if cls == 'sectionHeading':
            chapter += 1
        elif cls == 'noteHeading':
            kind = 'highlight' if body.startswith('Highlight') else 'note'
        elif kind == 'highlight':
            items.append({'chapter': chapter, 'highlight': ' '.join(body.split()), 'notes': []})
        else:
            items[-1]['notes'] += [line.strip() for line in body.splitlines() if line.strip()]
    return items


def letters(text):
    return re.sub(r'[^a-z]', '', text.lower())


def align(highlight, segments):
    """(segment ids the highlight covers, score) within one chapter's segments."""
    chap, owner = '', []
    for s in segments:
        t = letters(s['text'])
        chap += t
        owner += [s['id']] * len(t)
    h = letters(highlight)
    # Anchor on the longest exact run, then align the highlight against a window around it.
    m = difflib.SequenceMatcher(None, chap, h, autojunk=False).find_longest_match(0, len(chap), 0, len(h))
    lo = max(0, m.a - m.b - 20)
    window = chap[lo:m.a - m.b + len(h) + 20]
    blocks = [b for b in difflib.SequenceMatcher(None, window, h, autojunk=False).get_matching_blocks() if b.size >= 3]
    score = sum(b.size for b in blocks) / len(h)
    hits = {}
    for b in blocks:
        for i in range(lo + b.a, lo + b.a + b.size):
            hits[owner[i]] = hits.get(owner[i], 0) + 1
    size = {s['id']: len(letters(s['text'])) for s in segments}
    ids = sorted(i for i, n in hits.items() if n >= min(MIN_OVERLAP, size[i] / 2))
    return ids, score
