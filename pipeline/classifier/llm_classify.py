"""Sound/no-sound gate with an LLM (zero-shot or few-shot), scored against the labels.

    python pipeline/classifier/llm_classify.py --mode zeroshot --chapters 6
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from train_classifier import chapters  # noqa: E402

ZEROSHOT = """You are helping build a soundscape for a novel. Read the passage and decide whether the \
target sentence describes or clearly implies a sound a listener could hear, such as a noise, a voice \
raised in a shout or song, weather, an animal or an object being used. Ordinary dialogue and thoughts \
do not count. Answer with exactly one word: SOUND or NO_SOUND.

Previous sentence: {prev}
Target sentence: {text}"""

FEWSHOT = """You are helping build a soundscape for a novel. Below are numbered sentences from one \
chapter. List the numbers of the sentences that describe or clearly imply a sound a listener could hear, \
such as a noise, a voice raised in a shout or song, weather, an animal or an object being used. Ordinary \
dialogue and thoughts do not count. Most sentences have no sound.

Examples:
{examples}

Sentences:
{window}

Answer with the numbers of the sound sentences only, separated by commas, or NONE."""


def api_key(env_file='.env'):
    """SCI_LLM_API_KEY from the environment, else from .env."""
    if 'SCI_LLM_API_KEY' not in os.environ and os.path.exists(env_file):
        with open(env_file, encoding='utf-8') as f:
            for line in f:
                k, _, v = line.strip().partition('=')
                if k.strip() == 'SCI_LLM_API_KEY':
                    return v.strip().strip('"\'')
    return os.environ['SCI_LLM_API_KEY']


def chat(prompt, a, tries=3):
    req = urllib.request.Request(
        a.base_url.rstrip('/') + '/chat/completions',
        data=json.dumps({'model': a.model, 'temperature': 0, 'max_tokens': a.max_tokens,
                         'messages': [{'role': 'user', 'content': prompt + (' /no_think' if a.no_think else '')}]}).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {a.key}'})
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                choice = json.load(r)['choices'][0]
            break
        except OSError as e:
            if t == tries - 1:
                raise
            print(f'  retrying after {e}')
            time.sleep(5)
    text = re.sub(r'<think>.*?</think>', '', choice['message']['content'] or '', flags=re.S).strip()
    if not text and choice.get('finish_reason') == 'length':
        print('  empty reply: out of tokens (raise --max-tokens or use --no-think)')
    return text


def flat(text):
    return ' '.join(text.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['zeroshot', 'fewshot'], required=True)
    ap.add_argument('--chapters', default='6', help='chapters to label, e.g. 6 or 5-6')
    ap.add_argument('--train', default='1-4', help='chapters the few-shot examples come from')
    ap.add_argument('--model', default='qwen3.8')
    ap.add_argument('--base-url', default='https://ol.sci.pitt.edu')
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--max-tokens', type=int, default=4000, help='room for the answer, and any thinking before it')
    ap.add_argument('--no-think', action='store_true', help="add Qwen's /no_think switch so it answers directly")
    ap.add_argument('--id', default='hobbit')
    ap.add_argument('--out', default='data/processed')
    a = ap.parse_args()
    a.key = api_key()

    stem = os.path.join(a.out, a.id)
    with open(f'{stem}.segments.csv', newline='', encoding='utf-8') as f:
        segs = list(csv.DictReader(f))
    with open(f'{stem}.labels.csv', newline='', encoding='utf-8') as f:
        gold = {int(r['id']): int(r['sound']) for r in csv.DictReader(f)}
    todo = [s for s in segs if int(s['chapter']) in chapters(a.chapters)]
    pred, reply = {}, {}

    if a.mode == 'zeroshot':
        for n, s in enumerate(todo, 1):
            if n % 50 == 0:
                print(f'  {n}/{len(todo)}')
            i = int(s['id'])
            prev = flat(segs[i - 1]['text']) if i and segs[i - 1]['chapter'] == s['chapter'] else '(start of chapter)'
            reply[i] = chat(ZEROSHOT.format(prev=prev, text=flat(s['text'])), a)
            pred[i] = int('SOUND' in reply[i].upper() and 'NO' not in reply[i].upper())
    else:
        train = [s for s in segs if int(s['chapter']) in chapters(a.train) and int(s['id']) in gold]
        rng = random.Random(0)
        shots = rng.sample([s for s in train if gold[int(s['id'])]], 4) + \
            rng.sample([s for s in train if not gold[int(s['id'])]], 4)
        rng.shuffle(shots)
        examples = '\n'.join(f"{flat(s['text'])} -> {'SOUND' if gold[int(s['id'])] else 'NO_SOUND'}" for s in shots)
        for b in range(0, len(todo), a.window):
            window = todo[b:b + a.window]
            text = chat(FEWSHOT.format(examples=examples, window='\n'.join(f"[{s['id']}] {flat(s['text'])}"
                                                                           for s in window)), a)
            said = {int(n) for n in re.findall(r'\d+', text)}
            for s in window:
                i = int(s['id'])
                pred[i], reply[i] = int(i in said), text if s is window[0] else ''

    with open(f'{stem}.llm_{a.mode}.{a.model}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'chapter', 'sound', 'pred', 'reply'])
        w.writerows([s['id'], s['chapter'], gold.get(int(s['id']), ''), pred[int(s['id'])], reply[int(s['id'])]]
                    for s in todo)

    pairs = [(gold[i], p) for i, p in pred.items() if i in gold]
    tp = sum(g and p for g, p in pairs)
    prec, rec = tp / max(1, sum(p for _, p in pairs)), tp / max(1, sum(g for g, _ in pairs))
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    print(f'{a.mode} {a.model} on ch {a.chapters}: {len(pairs)} segments, {sum(p for _, p in pairs)} predicted sound, '
          f'P {prec:.2f} R {rec:.2f} F1 {f1:.2f}')


if __name__ == '__main__':
    main()
