# book-soundscapes

CS 1671 group project: give a book a soundscape by matching each sentence to a
captioned sound clip. This repo is the text side: book in, cleaned and
segmented text out, then each segment matched to AudioCaps captions.

## Run it

Python 3.9+. Book processing and TF-IDF use only the standard library; the
other methods need `pip install -r requirements-extra.txt` (and, for
`tfidf_pos`, `python -m nltk.downloader averaged_perceptron_tagger_eng`).

```
# 1. EPUB -> cleaned text, segments, e-reader JSON, annotation sheet
python scripts/process_book.py data/raw/hobbit.epub --id hobbit --title "The Hobbit" --annotation-sheet

# 2. match segments to captions (needs data/raw/audiocaps_clean.csv and audiocaps_clips.csv)
python scripts/match.py --method tfidf        # or tfidf_pos, lsa, minilm

python -m unittest discover -s tests
```

`minilm` also needs the model in `models/all-MiniLM-L6-v2/`
(https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz).

## Layout

| path | what |
|---|---|
| `soundscapes/epub.py` | reads paragraphs from the EPUB in reading order |
| `soundscapes/clean.py` | drops front matter, finds chapters, normalizes text, spots verse |
| `soundscapes/segment.py` | splits into sentences, applies the merge rule, assigns ids and offsets |
| `soundscapes/book.py` | runs the three above and writes the output files |
| `soundscapes/tfidf.py` | TF-IDF vectorizer and sparse nearest-neighbour search |
| `soundscapes/postag.py` | part-of-speech-tagged words for `tfidf_pos` |
| `soundscapes/minilm.py` | MiniLM sentence embeddings |
| `soundscapes/methods.py` | the four methods behind one interface, top-k search, cutoff |
| `scripts/process_book.py`, `scripts/match.py` | command-line entry points |

## Outputs (`data/processed/`)

| file | what |
|---|---|
| `<id>.txt` | cleaned text; all character offsets point into it |
| `<id>.segments.csv` / `.json` | one row per segment; the JSON is the e-reader format with `has_sound` and `clip` empty |
| `<id>.chapters.csv` | chapter titles and offsets |
| `<id>.annotation.csv` | segments with an empty `sound` column for hand labelling |
| `<id>.<method>.top.csv` | each segment's top 3 captions and cosine scores |
| `<id>.<method>.soundscape.json` | e-reader format with `has_sound` and `clip` filled in |

E-reader format:

```json
{"book": "The Hobbit",
 "segments": [
   {"id": 412, "char_start": 51230, "char_end": 51311, "text": "The thunder crashed ...",
    "has_sound": true, "clip": {"audiocaps_id": "91234", "start_sec": 30, "score": 0.61}},
   {"id": 413, "char_start": 51312, "char_end": 51360, "text": "Bilbo wished he were home.",
    "has_sound": false, "clip": null}]}
```

## Design decisions

**Data stays out of git.** The Hobbit is under copyright and is used only
inside the group, so `data/` and `models/` are git-ignored. Tests use a
public-domain excerpt of *The Princess and the Goblin*.

**Segments are sentences, merged if under 20 characters.** A sentence is the
unit a reader experiences a sound with. Very short pieces like `"Why?"` carry
too little to match, so they join the next sentence (or the previous one at a
chapter end). Each segment keeps character offsets into the cleaned text so
the e-reader can highlight it.

**Songs are kept but flagged.** Runs of three or more short, non-dialogue
lines are marked `is_verse`, so they can be labelled or skipped as a group.

**We match against captions, not audio.** AudioCaps gives each 10-second
AudioSet clip a human-written description. Comparing a sentence to a caption
is a text-to-text problem, which every method below can handle. Clips tagged
as speech or music are left out, since a soundscape should not talk over the
reader.

**TF-IDF is fitted on segments and captions together.** Cosine similarity is
only meaningful if both sides use the same vocabulary and the same idf
weights. Word pairs are included so "crackling fire" counts as a unit. It is
written in plain Python and uses the same weighting as scikit-learn.

**`tfidf_pos` adds part-of-speech tags.** TF-IDF cannot tell "he saw" from
"a hand saw". Tagging each word (`saw_v`, `saw_n`) separates them. It helps on
those cases, but the tagger often mis-tags short caption fragments.

**LSA reduces TF-IDF to 300 dimensions.** The idea is that words that occur
together (thunder, storm, rain) end up close, so a match no longer needs the
exact word. On our short texts it pushes all scores up and blurs meanings, and
its neighbours change between runs, so it is the weakest of the four.

**MiniLM is the dense baseline.** A small transformer trained so that
sentences with similar meanings get similar vectors. It reads each word in
context, so it is the most robust to word traps.

**Every method shares one pipeline.** Embed segments and captions, take each
segment's top 3 captions by cosine similarity, and give a segment a sound when
its best score clears a cutoff.

**The cutoff is set per method, as a share of segments.** Scores are not on
the same scale across methods (a TF-IDF 0.3 is roughly a MiniLM 0.47), so
`--share 0.12` picks each method's cutoff so 12% of segments get a sound.
That share is a placeholder. A sound/no-sound classifier trained on the
hand-labelled segments should replace it.
