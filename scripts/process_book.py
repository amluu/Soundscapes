"""Clean and segment an EPUB into the files listed in the README.

    python scripts/process_book.py BOOK.epub --id hobbit --title "The Hobbit" [--annotation-sheet]
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from soundscapes.book import odd_characters, process_epub, write_annotation_sheet, write_outputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('epub')
    ap.add_argument('--id', required=True, help='short book id, used for output file names')
    ap.add_argument('--title', default='')
    ap.add_argument('--out', default='data/processed')
    ap.add_argument('--annotation-sheet', action='store_true', help='also write <id>.annotation.csv')
    a = ap.parse_args()

    text, segs, chapters = process_epub(a.epub)
    files = write_outputs(a.out, a.id, a.title, text, segs, chapters)
    if a.annotation_sheet:
        files.append(os.path.join(a.out, f'{a.id}.annotation.csv'))
        write_annotation_sheet(files[-1], segs)

    n = sorted(s['n_tokens'] for s in segs)
    print(f'{len(chapters)} chapters, {len(segs)} segments ({sum(s["is_verse"] for s in segs)} verse), '
          f'{len(text):,} chars, {sum(n):,} tokens; tokens/segment median {n[len(n) // 2]}, max {n[-1]}')
    for c in chapters:
        print(f'  ch {c["chapter"]:>2}  {c["title"]}')
    odd = odd_characters(text)
    if odd:
        print('unusual characters:', dict(collections.Counter(c for c, _, _ in odd)))
        for c, pos, ctx in odd[:20]:
            print(f'  {c!r} @{pos}: ...{ctx}...')
    print('wrote', *files, sep='\n  ')


if __name__ == '__main__':
    main()
