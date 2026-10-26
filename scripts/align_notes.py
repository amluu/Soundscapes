"""Turn a Kindle notebook export of sound annotations into per-segment labels.

    python scripts/align_notes.py data/raw/natalie_notes.html

Writes <id>.labels.csv (sound = 1 where a highlight covers the segment) and
<id>.alignment.csv (where each highlight landed).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from soundscapes.notes import MIN_SCORE, align, read_notebook  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('notebook')
    ap.add_argument('--id', default='hobbit')
    ap.add_argument('--out', default='data/processed')
    a = ap.parse_args()

    stem = os.path.join(a.out, a.id)
    with open(f'{stem}.segments.csv', newline='', encoding='utf-8') as f:
        segs = [{**r, 'id': int(r['id']), 'chapter': int(r['chapter'])} for r in csv.DictReader(f)]
    by_chapter = defaultdict(list)
    for s in segs:
        by_chapter[s['chapter']].append(s)

    items = read_notebook(a.notebook)
    notes, failed = defaultdict(list), []
    with open(f'{stem}.alignment.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['n', 'chapter', 'segment_ids', 'score', 'notes', 'highlight'])
        for n, it in enumerate(items, 1):
            ids, score = align(it['highlight'], by_chapter[it['chapter']])
            if score < MIN_SCORE or not ids:
                failed.append(n)
                ids = []
            for i in ids:
                notes[i] += [x for x in it['notes'] if x not in notes[i]]
            w.writerow([n, it['chapter'], ' '.join(map(str, ids)), round(score, 3), ' | '.join(it['notes']),
                        it['highlight']])

    chapters = sorted({it['chapter'] for it in items})
    rows = [s for s in segs if s['chapter'] in chapters]
    with open(f'{stem}.labels.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'chapter', 'sound', 'description'])
        for s in rows:
            w.writerow([s['id'], s['chapter'], int(s['id'] in notes), '; '.join(notes.get(s['id'], []))])

    print(f'{len(items)} highlights in chapters {chapters[0]}-{chapters[-1]}, {len(failed)} not aligned {failed or ""}')
    print(f'{len(notes)} of {len(rows)} segments labelled sound ({len(notes) / len(rows):.1%})')


if __name__ == '__main__':
    main()
