"""Turn raw EPUB paragraphs into cleaned chapters of prose and verse blocks."""
import re
from dataclasses import dataclass, field

CHAPTER_RE = re.compile(r'^chapter\s+([0-9]+|[ivxlc]+|[a-z-]+)\s*$', re.I)
STOP_RE = re.compile(r'^(table of contents|contents|appendix|about the author)\s*$', re.I)

QUOTES = {'\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'", '\xab': '"', '\xbb': '"'}
DASHES = {'\u2013': '-', '\u2014': ' - ', '\u2015': ' - '}

VERSE_MAX_CHARS = 70
VERSE_MIN_LINES = 3  # this many short lines in a row is a song or poem


@dataclass
class Block:
    """A prose paragraph or a verse block (lines joined by newlines)."""
    text: str
    is_verse: bool = False


@dataclass
class Chapter:
    number: int
    label: str
    title: str
    blocks: list = field(default_factory=list)


def normalize(text):
    """Straighten quotes and dashes, collapse whitespace."""
    for a, b in {**QUOTES, **DASHES}.items():
        text = text.replace(a, b)
    text = text.replace('\xa0', ' ').replace('\xad', '')
    return re.sub(r'\s+', ' ', text).strip()


def _is_verse_line(p):
    return len(p) <= VERSE_MAX_CHARS and p.count('"') <= 1 and not p.endswith(':')


def _ends_open(p):
    """True if the paragraph stops mid-sentence."""
    return bool(re.search(r'[A-Za-z,]$', p))


def to_blocks(paras):
    """Group paragraphs into prose and verse blocks, rejoining split sentences."""
    paras = [normalize(p) for p in paras]
    paras = [p for p in paras if p]
    verse = [False] * len(paras)
    i = 0
    while i < len(paras):
        j = i
        while j < len(paras) and _is_verse_line(paras[j]):
            j += 1
        if j - i >= VERSE_MIN_LINES:
            verse[i:j] = [True] * (j - i)
        i = max(j, i + 1)

    blocks = []
    for p, v in zip(paras, verse):
        prev = blocks[-1] if blocks else None
        if v and prev and prev.is_verse:
            prev.text += '\n' + p
        elif (not v and prev and not prev.is_verse and _ends_open(prev.text)
              and re.match(r'["\']?[a-z]', p)):
            prev.text += ' ' + p
        else:
            blocks.append(Block(p, v))
    return blocks


def _roman(s):
    vals = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100}
    total = 0
    for a, b in zip(s.lower(), s.lower()[1:] + ' '):
        total += -vals[a] if vals.get(b, 0) > vals[a] else vals[a]
    return total


def chapters_from_paragraphs(paras):
    """Split paragraphs into chapters, dropping front matter and anything from the table of contents on."""
    chapters, expect_title = [], False
    for raw in paras:
        p = normalize(raw)
        m = CHAPTER_RE.match(p)
        if m:
            tok = m.group(1)
            num = int(tok) if tok.isdigit() else (_roman(tok) if re.fullmatch(r'[ivxlc]+', tok, re.I) else len(chapters) + 1)
            chapters.append(Chapter(num, p, ''))
            expect_title = True
            continue
        if not chapters:
            continue
        if STOP_RE.match(p):
            break
        if expect_title:
            chapters[-1].title, expect_title = p, False
            continue
        chapters[-1].blocks.append(p)
    for ch in chapters:
        ch.blocks = to_blocks(ch.blocks)
    return chapters
