"""EPUB in; cleaned text, segment table, e-reader JSON and annotation sheet out."""
import csv
import json
import os
import re

from .clean import chapters_from_paragraphs
from .epub import epub_paragraphs
from .segment import build

CSV_COLS = ['id', 'chapter', 'paragraph', 'char_start', 'char_end', 'is_verse', 'n_chars', 'n_tokens', 'text']


def process_epub(path):
    paras = [p for _, ps in epub_paragraphs(path) for p in ps]
    chapters = chapters_from_paragraphs(paras)
    text, segments, chapter_rows = build(chapters)
    for s in segments:
        assert text[s['char_start']:s['char_end']] == s['text']
    return text, segments, chapter_rows


def odd_characters(text, context=30):
    """Characters outside plain printable ASCII, with a little context, for eyeballing."""
    out = []
    for m in re.finditer(r'[^\x20-\x7e\n]', text):
        out.append((m.group(), m.start(), text[max(0, m.start() - context):m.end() + context].replace('\n', ' / ')))
    return out


def write_outputs(out_dir, book_id, title, text, segments, chapter_rows):
    """Write <id>.txt, <id>.segments.csv, <id>.segments.json and <id>.chapters.csv."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, book_id)
    with open(f'{stem}.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    with open(f'{stem}.segments.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, CSV_COLS)
        w.writeheader()
        w.writerows({k: s[k] for k in CSV_COLS} for s in segments)
    # E-reader format; scripts/match.py fills in has_sound and clip.
    doc = {
        'book': title or book_id,
        'segments': [{**{k: s[k] for k in ('id', 'char_start', 'char_end', 'text')},
                      'has_sound': None, 'clip': None} for s in segments],
    }
    with open(f'{stem}.segments.json', 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    with open(f'{stem}.chapters.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['chapter', 'label', 'title', 'char_start', 'char_end'])
        w.writeheader()
        w.writerows(chapter_rows)
    return [f'{stem}.txt', f'{stem}.segments.csv', f'{stem}.segments.json', f'{stem}.chapters.csv']


def write_annotation_sheet(path, segments):
    """One row per segment with an empty label column (1 = sound, 0 = no sound)."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'chapter', 'is_verse', 'text', 'sound', 'notes'])
        for s in segments:
            w.writerow([s['id'], s['chapter'], int(s['is_verse']), s['text'], '', ''])
