# book-soundscapes

CS 1671 group project: Book Soundscapes 

Ava Luu, Aidan Robin, Nitin Chandran, Natalie Goldsworthy

## Run

### Requirements
Python 3.9+. 
Run: 
`pip install -r requirements.txt` 
`python -m nltk.downloader averaged_perceptron_tagger_eng`.

### 1. EPUB -> cleaned text, segments, e-reader JSON, annotation sheet
```
python pipeline/preprocessing/process_book.py data/raw/hobbit.epub --id hobbit --title "The Hobbit" --annotation-sheet
```

### 2. match segments to captions (needs data/raw/audiocaps_clean.csv and audiocaps_clips.csv)
```
python pipeline/matching/match.py --method tfidf        
# or tfidf_pos, lsa, minilm
```

### 3. hand labels (Kindle notebook export) -> per-segment labels, then the sound/no-sound classifier
```
python pipeline/preprocessing/align_notes.py data/raw/natalie_notes.html
python pipeline/classifier/train_classifier.py --features tfidf --train 1-4 --dev 5 --test 6   
# or --features minilm

python pipeline/matching/match.py --method minilm --gate data/processed/hobbit.minilm.gate.csv
```

### 4. LLM gate, scored against the labels
```
python pipeline/classifier/llm_classify.py --mode zeroshot 
# needs SCI_LLM_API_KEY in .env; also --mode fewshot
```

### 5. manual eval: every matcher behind the gate, then a blind sheet to grade
```
python pipeline/matching/match.py --method tfidf --gate data/processed/hobbit.minilm.gate.csv   
# and the other three

python pipeline/manual-eval/export_eval.py --chapters 7-19 --kept 100 --rejected 50

python pipeline/manual-eval/score_eval.py data/processed/hobbit.eval.graded.*.csv
# one sheet per grader

python pipeline/manual-eval/score_gate_marks.py hobbit.gate_marks.*.csv   # group's real-sound marks on the two gates
```

`minilm` also needs the model in `models/all-MiniLM-L6-v2/`
(https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz).

## Layout

| path | what |
|---|---|
| `pipeline/preprocessing/` | `epub.py` reads the EPUB, `clean.py` drops front matter, finds chapters and spots verse, `segment.py` splits sentences and assigns ids and offsets, `book.py` runs them and writes the files; `notes.py` aligns Kindle highlights to segments. Scripts: `process_book.py`, `align_notes.py` |
| `pipeline/matching/` | `tfidf.py` and `postag.py` (TF-IDF, plain and POS-tagged), `minilm.py` (MiniLM embeddings), `methods.py` (the four methods, top-k search, cutoff). Script: `match.py` |
| `pipeline/classifier/` | `train_classifier.py` (logistic regression gate), `llm_classify.py` (LLM gate) |
| `pipeline/manual-eval/` | `export_eval.py` (blind grading sheet), `score_eval.py` (tally by method, grader kappa), `score_gate_marks.py` (group's marks on the two gates) |
| `output/` | soundscape JSONs for the frontend |

## Outputs (`data/processed/`)

| file | what |
|---|---|
| `<id>.txt` | cleaned text; all character offsets point into it |
| `<id>.segments.csv` / `.json` | one row per segment; the JSON is the e-reader format with `has_sound` and `clip` empty |
| `<id>.chapters.csv` | chapter titles and offsets |
| `<id>.annotation.csv` | segments with an empty `sound` column for hand labelling |
| `<id>.<method>.top.csv` | each segment's top 3 captions and cosine scores |
| `<id>.labels.csv` / `<id>.alignment.csv` | sound labels and descriptions per segment; where each highlight landed |
| `<id>.<features>.gate.csv` | the classifier's `p_sound` and `has_sound` for every segment |
| `<id>.<features>.lr_eval.csv`, `<id>.tfidf.lr_terms.csv` | classifier predictions on labelled chapters; top terms |
| `<id>.eval.csv` / `.json`, `<id>.eval_key.csv` | items to grade (methods hidden, shuffled); which methods made each item |

## Soundscapes (`output/`)

The e-reader JSONs, committed so the frontend can read them.

| file | what |
|---|---|
| `<id>.<method>.soundscape.json` | `has_sound` and `clip` filled in by the score cutoff |
| `<id>.<method>.gated.soundscape.json` | sounds only where the gate says so (`hobbit.minilm.gated` is the full-book demo) |

Format:

```json
{"book": "The Hobbit",
 "segments": [
   {"id": 412, "char_start": 51230, "char_end": 51311, "text": "The thunder crashed ...",
    "has_sound": true, "clip": {"audiocaps_id": "91234", "start_sec": 30, "score": 0.61,
             "ytid": "IU9Z_7GsYNM", "caption": "Thunder roars"}},
   {"id": 413, "char_start": 51312, "char_end": 51360, "text": "Bilbo wished he were home.",
    "has_sound": false, "clip": null}]}
```

## Notes

* Segments are sentences, merged if under 20 characters.

* Songs are kept but flagged: three or more short, non-dialogue
lines are marked as `is_verse`, so they can be labelled or skipped.

* We match against the human-written description in AudioCaps, not the actual audio (text to text)

* TF-IDF is fitted on segments and captions together.

* `tfidf_pos` adds part-of-speech tags.

* LSA reduces TF-IDF to 300 dimensions. The hypothesis is that words that occur together (thunder, storm, rain) end up close, so a match no longer needs the exact word. On our short texts it pushes all scores up and blurs meanings, and its neighbours change between runs, so it is the weakest of the four.

* MiniLM is used as a dense baseline.

* Every method shares one pipeline, but the cutoff is set per method, either as a % of what segments we want labelled, or gated by a classifier.

* The classifier is logistic regression, split by chapter.
