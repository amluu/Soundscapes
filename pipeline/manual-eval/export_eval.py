"""Export a blind, shuffled sheet for grading the gated soundscapes by hand.

    python pipeline/manual-eval/export_eval.py --chapters 7-19 --kept 100 --rejected 50
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'classifier'))
from train_classifier import chapters  # noqa: E402

COLS = ['item', 'segment_id', 'chapter', 'previous', 'text', 'has_sound', 'audiocaps_id', 'youtube_id',
        'start_sec', 'url', 'caption', 'grade', 'note']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', default=['tfidf', 'tfidf_pos', 'lsa', 'minilm'])
    ap.add_argument('--chapters', default='7-19')
    ap.add_argument('--kept', type=int, default=100)
    ap.add_argument('--rejected', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--id', default='hobbit')
    ap.add_argument('--out', default='data/processed')
    ap.add_argument('--soundscapes', default='output', help='folder with the gated soundscape JSONs')
    ap.add_argument('--captions', default='data/raw/audiocaps_clean.csv')
    a = ap.parse_args()

    stem = os.path.join(a.out, a.id)
    with open(f'{stem}.segments.csv', newline='', encoding='utf-8') as f:
        segs = list(csv.DictReader(f))
    with open(a.captions, newline='', encoding='utf-8') as f:
        caps = {r['audiocap_id']: r for r in csv.DictReader(f)}
    picks = defaultdict(lambda: defaultdict(list))  # segment id -> audiocaps id -> methods
    for m in a.methods:
        with open(os.path.join(a.soundscapes, f'{a.id}.{m}.gated.soundscape.json'), encoding='utf-8') as f:
            for s in json.load(f)['segments']:
                if s['has_sound']:
                    picks[s['id']][s['clip']['audiocaps_id']].append(m)

    pool = [int(s['id']) for s in segs if int(s['chapter']) in chapters(a.chapters)]
    rng = random.Random(a.seed)
    kept = rng.sample([i for i in pool if i in picks], a.kept)
    rejected = rng.sample([i for i in pool if i not in picks], a.rejected)

    items = [(i, cap_id, methods) for i in kept for cap_id, methods in picks[i].items()]
    items += [(i, None, ['gate rejected']) for i in rejected]
    rng.shuffle(items)
    rows, key = [], []
    for n, (i, cap_id, methods) in enumerate(items, 1):
        s, c = segs[i], caps.get(cap_id, {})
        prev = segs[i - 1]['text'] if i and segs[i - 1]['chapter'] == s['chapter'] else ''
        rows.append({'item': n, 'segment_id': i, 'chapter': s['chapter'], 'previous': prev, 'text': s['text'],
                     'has_sound': int(cap_id is not None), 'audiocaps_id': cap_id or '',
                     'youtube_id': c.get('youtube_id', ''), 'start_sec': c.get('start_time', ''),
                     'url': f"https://www.youtube.com/watch?v={c['youtube_id']}&t={c['start_time']}s" if c else '',
                     'caption': c.get('caption', ''), 'grade': '', 'note': ''})
        key.append([n, ' '.join(methods)])

    with open(f'{stem}.eval.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows)
    with open(f'{stem}.eval.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    with open(f'{stem}.eval_key.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['item', 'methods'])
        w.writerows(key)
    print(f'{len(rows)} items: {a.kept} kept segments ({len(rows) - a.rejected} clip picks) '
          f'and {a.rejected} rejected segments, chapters {a.chapters}')


if __name__ == '__main__':
    main()
