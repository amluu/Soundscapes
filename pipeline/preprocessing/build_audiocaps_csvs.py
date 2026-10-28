#!/usr/bin/env python3
"""
Build data/raw/audiocaps_clean.csv and data/raw/audiocaps_clips.csv for
pipeline/matching/match.py.

audiocaps_clean.csv   audiocap_id, youtube_id, start_time, caption
    Every AudioCaps caption (train + val + test by default), whitespace
    normalised, empty captions dropped, and exact duplicate captions for the
    same clip dropped.

audiocaps_clips.csv   youtube_id, start_time, speech_or_music, has_speech,
                      has_music, label_source, audioset_labels
    One row per clip. speech_or_music is "True" when the clip's AudioSet labels
    include Speech or Music (or any of their sub-classes), which is what
    match.py uses to drop noisy clips. If a clip is missing from AudioSet, a
    caption keyword check is used instead (label_source = "caption").

Usage (from the repo root):
    python build_audiocaps_csvs.py
    python build_audiocaps_csvs.py --audiocaps-version 2
    python build_audiocaps_csvs.py --csv-dir path/to/audiocaps_csvs
    # after downloading audio, keep only clips whose mp3 exists:
    python build_audiocaps_csvs.py --audio-dir frontend_evaluation/ytid

Sources (fetched automatically unless local copies are given):
    AudioCaps   https://github.com/cdjkim/audiocaps   (dataset/ or dataset2.0/)
    AudioSet    https://research.google.com/audioset/download.html
                (segment CSVs + ontology.json; unbalanced_train is ~100 MB)
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import urllib.request

AUDIOCAPS_URL = {
    "1": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/{split}.csv",
    "2": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset2.0/{split}.csv",
}
AUDIOSET_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{name}.csv"
AUDIOSET_FILES = ["eval_segments", "balanced_train_segments", "unbalanced_train_segments"]
ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"

SPEECH_ID = "/m/09x0r"  # Speech
MUSIC_ID = "/m/04rlf"   # Music

# Fallback only, for clips not found in AudioSet.
SPEECH_WORDS = re.compile(r"\b(speak|speaks|speaking|speech|talk|talks|talking|narrat\w*|"
                          r"conversation|announc\w*|says|said|voice|voices|whisper\w*|"
                          r"shout\w*|yell\w*|laugh\w*|sing|sings|singing|chant\w*)\b", re.I)
MUSIC_WORDS = re.compile(r"\b(music|musical|song|songs|melody|guitar|piano|drum|drums|"
                         r"violin|orchestra|band|instrument\w*|tune|beat)\b", re.I)


def fetch_text(url):
    print(f"  fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def read_source(local_dir, filename, url):
    if local_dir:
        path = os.path.join(local_dir, filename)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
        if not url:
            sys.exit(f"missing {path}")
    return fetch_text(url)


def descendants(ontology, root):
    """root plus every class under it in the AudioSet ontology."""
    children = {n["id"]: n.get("child_ids", []) for n in ontology}
    out, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n not in out:
            out.add(n)
            stack.extend(children.get(n, []))
    return out


def load_audiocaps(splits, version, csv_dir):
    rows, seen, seen_ids, dropped = [], set(), set(), 0
    for split in splits:
        text = read_source(csv_dir, f"{split}.csv",
                           None if csv_dir else AUDIOCAPS_URL[version].format(split=split))
        for r in csv.DictReader(io.StringIO(text)):
            cap = " ".join(r["caption"].split())
            ytid = r["youtube_id"].strip()
            key = (ytid, cap.lower().rstrip("."))
            if not cap or key in seen or r["audiocap_id"] in seen_ids:
                dropped += 1
                continue
            seen.add(key)
            seen_ids.add(r["audiocap_id"])
            rows.append({"audiocap_id": r["audiocap_id"].strip(), "youtube_id": ytid,
                         "start_time": int(float(r["start_time"])), "caption": cap})
    print(f"AudioCaps: {len(rows)} captions kept, {dropped} empty/duplicate dropped", file=sys.stderr)
    return rows


def load_audioset_labels(ytids, audioset_dir):
    """{ytid: set(label ids)} for the ytids we care about."""
    labels = {}
    for name in AUDIOSET_FILES:
        text = read_source(audioset_dir, f"{name}.csv",
                           None if audioset_dir else AUDIOSET_URL.format(name=name))
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            ytid, _, rest = line.partition(",")
            if ytid in ytids:
                # YTID, start_seconds, end_seconds, "label1,label2,..."
                m = re.search(r'"([^"]*)"', rest)
                labels[ytid] = set(m.group(1).split(",")) if m else set()
    return labels


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--audiocaps-version", choices=["1", "2"], default="1",
                   help="1 = dataset/, 2 = dataset2.0/ (ignored with --csv-dir)")
    p.add_argument("--csv-dir", help="local folder with AudioCaps train/val/test.csv")
    p.add_argument("--audioset-dir", help="local folder with the AudioSet segment CSVs and ontology.json")
    p.add_argument("--audio-dir", help="only keep clips whose <ytid>.mp3 exists here")
    a = p.parse_args()

    caps = load_audiocaps(a.splits, a.audiocaps_version, a.csv_dir)

    if a.audio_dir:
        have = {f[:-4] for f in os.listdir(a.audio_dir) if f.endswith(".mp3")}
        before = len(caps)
        caps = [c for c in caps if c["youtube_id"] in have]
        print(f"--audio-dir: kept {len(caps)}/{before} captions with an mp3", file=sys.stderr)

    clips = {}
    for c in caps:
        clips.setdefault(c["youtube_id"], {"start_time": c["start_time"], "captions": []})
        clips[c["youtube_id"]]["captions"].append(c["caption"])

    ontology = json.loads(read_source(a.audioset_dir, "ontology.json",
                                      None if a.audioset_dir else ONTOLOGY_URL))
    speech_ids, music_ids = descendants(ontology, SPEECH_ID), descendants(ontology, MUSIC_ID)
    names = {n["id"]: n["name"] for n in ontology}
    labels = load_audioset_labels(set(clips), a.audioset_dir)

    os.makedirs(a.out_dir, exist_ok=True)
    clean_path = os.path.join(a.out_dir, "audiocaps_clean.csv")
    with open(clean_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audiocap_id", "youtube_id", "start_time", "caption"])
        w.writeheader()
        w.writerows(caps)

    clips_path = os.path.join(a.out_dir, "audiocaps_clips.csv")
    n_noisy = n_fallback = 0
    with open(clips_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["youtube_id", "start_time", "speech_or_music", "has_speech", "has_music",
                    "label_source", "audioset_labels"])
        for ytid, info in clips.items():
            if ytid in labels:
                ls = labels[ytid]
                speech, music, src = bool(ls & speech_ids), bool(ls & music_ids), "audioset"
            else:
                text = " ".join(info["captions"])
                speech, music, src = bool(SPEECH_WORDS.search(text)), bool(MUSIC_WORDS.search(text)), "caption"
                ls = set()
                n_fallback += 1
            n_noisy += speech or music
            w.writerow([ytid, info["start_time"], str(speech or music), str(speech), str(music), src,
                        "; ".join(sorted(names.get(l, l) for l in ls))])

    print(f"wrote {clean_path} ({len(caps)} captions)")
    print(f"wrote {clips_path} ({len(clips)} clips, {n_noisy} speech_or_music=True, "
          f"{n_fallback} labelled from captions because not found in AudioSet)")


if __name__ == "__main__":
    main()
