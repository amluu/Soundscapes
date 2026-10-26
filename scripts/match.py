"""Match book segments to AudioCaps captions with one method, then apply a score cutoff.

    python scripts/match.py --method minilm --share 0.12

Steps: embed segments and captions -> top-3 captions per segment by cosine
similarity -> segments whose best score clears the cutoff get that clip.
--share picks the cutoff that gives that share of segments a sound; --cutoff sets it.
--gate uses the classifier from scripts/train_classifier.py instead of a cutoff.

Writes to --out (default data/processed):
  <id>.<method>.top.csv          each segment's top 3 captions and scores
  <method>.vectorizer.json       fitted vocabulary and idf (TF-IDF methods only)
and to --json-out (default output), for the e-reader:
  <id>.<method>.soundscape.json  has_sound and clip filled in (<id>.<method>.gated.soundscape.json with --gate)
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from soundscapes.methods import METHODS, cutoff_for_share, top_k  # noqa: E402


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', choices=METHODS, required=True)
    ap.add_argument('--id', default='hobbit')
    ap.add_argument('--out', default='data/processed')
    ap.add_argument('--json-out', default='output')
    ap.add_argument('--captions', default='data/raw/audiocaps_clean.csv')
    ap.add_argument('--clips', default='data/raw/audiocaps_clips.csv', help="'' to keep speech and music clips")
    ap.add_argument('--model', default='models/all-MiniLM-L6-v2', help='MiniLM model folder')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--share', type=float, default=0.12)
    g.add_argument('--cutoff', type=float)
    g.add_argument('--gate', help='<id>.<features>.gate.csv from train_classifier.py')
    a = ap.parse_args()

    stem = os.path.join(a.out, f'{a.id}')
    with open(f'{stem}.segments.json', encoding='utf-8') as f:
        book = json.load(f)
    caps = read_csv(a.captions)
    if a.clips:
        noisy = {r['youtube_id'] for r in read_csv(a.clips) if r['speech_or_music'] == 'True'}
        caps = [c for c in caps if c['youtube_id'] not in noisy]
    segs = book['segments']
    print(f'{a.method}: {len(segs)} segments, {len(caps)} captions')

    S, C, extra = METHODS[a.method]([s['text'] for s in segs], [c['caption'] for c in caps], model_dir=a.model)
    matches = top_k(S, C)
    if 'vectorizer' in extra:
        extra['vectorizer'].save(os.path.join(a.out, f'{a.method}.vectorizer.json'))

    with open(f'{stem}.{a.method}.top.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['segment_id', 'rank', 'score', 'audiocaps_id', 'youtube_id', 'start_sec', 'caption', 'text'])
        for s, m in zip(segs, matches):
            for rank, (i, score) in enumerate(m, 1):
                c = caps[i]
                w.writerow([s['id'], rank, round(score, 4), c['audiocap_id'], c['youtube_id'],
                            c['start_time'], c['caption'], s['text']])

    if a.gate:
        gate = {int(r['id']): r['has_sound'] == '1' for r in read_csv(a.gate)}
        keep = [gate[s['id']] for s in segs]
        print(f'gate: {a.gate}')
    else:
        cutoff = a.cutoff if a.cutoff is not None else cutoff_for_share(matches, a.share)
        keep = [bool(m) and m[0][1] >= cutoff for m in matches]
        print(f'cutoff {cutoff:.3f}')
    for s, m, k in zip(segs, matches, keep):
        s['has_sound'] = k and bool(m)
        c = caps[m[0][0]] if s['has_sound'] else None
        s['clip'] = {'audiocaps_id': c['audiocap_id'], 'start_sec': float(c['start_time']),
                     'score': round(m[0][1], 4), 'ytid': c['youtube_id'], 'caption': c['caption']} if c else None
    os.makedirs(a.json_out, exist_ok=True)
    out = os.path.join(a.json_out, f"{a.id}.{a.method}{'.gated' if a.gate else ''}.soundscape.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(book, f, indent=1, ensure_ascii=False)
    n = sum(s['has_sound'] for s in segs)
    print(f'{n} segments ({n / len(segs):.1%}) get a sound')


if __name__ == '__main__':
    main()
