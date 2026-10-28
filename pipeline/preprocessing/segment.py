"""Join the cleaned chapters into one text and split it into segments with character offsets."""
import re

MIN_CHARS = 20
VERSE_LINES_PER_SEGMENT = 4
ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'st', 'mt', 'co', 'vs', 'etc', 'ft', 'no'}

_BOUNDARY = re.compile(r'[.!?]+["\')\]]*(?=\s+["\'(]?[A-Z0-9])')
_SENT_END = re.compile(r'[.!?]["\')\]]*$')


def sentence_spans(text):
    """[start, end) spans of the sentences in a prose paragraph."""
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        word = re.search(r'(\w+)\W*$', text[start:m.start() + 1])
        if word and m.group().startswith('.') and word.group(1).lower() in ABBREVIATIONS:
            continue
        spans.append((start, m.end()))
        start = m.end()
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        spans.append((start, len(text.rstrip())))
    return spans


def verse_spans(text):
    """Split a verse block at sentence ends or every few lines."""
    spans, start, n, pos = [], 0, 0, 0
    for line in text.split('\n'):
        end = pos + len(line)
        n += 1
        if _SENT_END.search(line) or n >= VERSE_LINES_PER_SEGMENT:
            spans.append((start, end))
            start, n = end + 1, 0
        pos = end + 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def build(chapters):
    """(text, segments, chapter_rows) for a list of clean.Chapter."""
    text, segments, chapter_rows = '', [], []
    for ci, ch in enumerate(chapters):
        if ci:
            text += '\n\n\n'
        ch_start = len(text)
        spans = []  # (abs_start, abs_end, block_index, is_verse)
        for bi, block in enumerate(ch.blocks):
            if bi:
                text += '\n\n'
            off = len(text)
            text += block.text
            fn = verse_spans if block.is_verse else sentence_spans
            spans += [(off + s, off + e, bi, block.is_verse) for s, e in fn(block.text)]

        merged, pending = [], None
        for s, e, bi, v in spans:
            if pending:
                s, bi, v = pending[0], pending[2], pending[3] or v
            if len(text[s:e]) < MIN_CHARS:
                pending = (s, e, bi, v)
                continue
            pending = None
            merged.append([s, e, bi, v])
        if pending:
            if merged:
                merged[-1][1] = pending[1]
                merged[-1][3] = merged[-1][3] or pending[3]
            else:
                merged.append(list(pending))

        for s, e, bi, v in merged:
            seg = text[s:e]
            segments.append({
                'id': len(segments),
                'chapter': ch.number,
                'char_start': s,
                'char_end': e,
                'text': seg,
                'paragraph': bi,
                'is_verse': v,
                'n_chars': len(seg),
                'n_tokens': len(seg.split()),
            })
        chapter_rows.append({'chapter': ch.number, 'label': ch.label, 'title': ch.title,
                             'char_start': ch_start, 'char_end': len(text)})
    return text, segments, chapter_rows
