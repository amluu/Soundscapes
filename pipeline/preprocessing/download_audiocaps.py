#!/usr/bin/env python3
"""
Download AudioCaps clips as 10-second MP3s named [ytid].mp3.

Each clip starts at the AudioCaps `start_time` and runs for 10 seconds.
Clips already in the output folder are skipped, so the script can be rerun
to resume or retry failures.

Requirements: Python 3.8+, ffmpeg on PATH, and yt-dlp (pip install -U yt-dlp)

Usage:
    # only the clips the frontend actually uses:
    python download_audiocaps.py --out frontend_evaluation/ytid --soundscapes output/*.soundscape.json
    # every AudioCaps clip:
    python download_audiocaps.py --out frontend_evaluation/ytid
    python download_audiocaps.py --out frontend_evaluation/ytid --splits test --workers 8
    python download_audiocaps.py --out frontend_evaluation/ytid --csv-dir ./audiocaps_csvs
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_URL = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/{split}.csv"
CLIP_SECONDS = 10


def load_clips(splits, csv_dir):
    """Return {youtube_id: start_time} across the requested splits."""
    clips, conflicts = {}, []
    for split in splits:
        if csv_dir:
            path = os.path.join(csv_dir, f"{split}.csv")
            text = open(path, encoding="utf-8").read()
        else:
            with urllib.request.urlopen(CSV_URL.format(split=split)) as r:
                text = r.read().decode("utf-8")
        for row in csv.DictReader(text.splitlines()):
            ytid, start = row["youtube_id"].strip(), int(float(row["start_time"]))
            if ytid in clips and clips[ytid] != start:
                conflicts.append((ytid, clips[ytid], start))
                continue
            clips[ytid] = start
    if conflicts:
        print(f"WARNING: {len(conflicts)} ytids have more than one start_time; "
              f"kept the first. Examples: {conflicts[:5]}", file=sys.stderr)
    return clips


def load_soundscape_clips(paths):
    """Return {ytid: start_sec} for every clip used in soundscape JSONs."""
    clips, conflicts = {}, []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for seg in data.get("segments", []):
            clip = seg.get("clip")
            if not clip or not clip.get("ytid"):
                continue
            ytid, start = clip["ytid"], int(float(clip["start_sec"]))
            if ytid in clips and clips[ytid] != start:
                conflicts.append((ytid, clips[ytid], start))
                continue
            clips[ytid] = start
    if conflicts:
        print(f"WARNING: {len(conflicts)} ytids have more than one start_sec; "
              f"kept the first. Examples: {conflicts[:5]}", file=sys.stderr)
    return clips


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def cut_clip(source, start, dest):
    """Cut [start, start+10s] from source (URL or file) into an MP3 at dest."""
    tmp = dest + ".part.mp3"
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
           "-ss", str(start), "-i", source, "-t", str(CLIP_SECONDS),
           "-vn", "-acodec", "libmp3lame", "-q:a", "2", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, (r.stderr.strip().splitlines() or ["ffmpeg failed"])[-1]
    os.replace(tmp, dest)  # atomic: a half-written file never gets the final name
    d = duration(dest)
    if d < CLIP_SECONDS - 0.5:
        return True, f"short clip ({d:.1f}s): video ends before start+10s"
    return True, ""


def download_one(ytid, start, out_dir, cookies):
    dest = os.path.join(out_dir, f"{ytid}.mp3")
    if os.path.exists(dest):
        return ytid, "skipped", ""
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
           "-f", "bestaudio/best", "-g", f"https://www.youtube.com/watch?v={ytid}"]
    if cookies:
        cmd[3:3] = ["--cookies", cookies]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return ytid, "failed", "yt-dlp timed out"
    url = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    if r.returncode != 0 or not url:
        msg = (r.stderr.strip().splitlines() or ["yt-dlp failed"])[-1]
        return ytid, "failed", msg
    try:
        ok, note = cut_clip(url, start, dest)
    except subprocess.TimeoutExpired:
        return ytid, "failed", "ffmpeg timed out"
    return ytid, ("ok" if ok else "failed"), note


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder, e.g. frontend_evaluation/ytid")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--soundscapes", nargs="+",
                   help="soundscape JSONs (e.g. output/*.soundscape.json); download only the clips they use")
    p.add_argument("--csv-dir", help="folder with train.csv/val.csv/test.csv (default: fetch from GitHub)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cookies", help="cookies.txt for YouTube if you hit bot/age checks")
    p.add_argument("--limit", type=int, help="only process the first N clips (for a trial run)")
    a = p.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")
    if subprocess.run([sys.executable, "-m", "yt_dlp", "--version"],
                      capture_output=True).returncode != 0:
        sys.exit("yt-dlp not installed: pip install -U yt-dlp")

    os.makedirs(a.out, exist_ok=True)
    if a.soundscapes:
        clips = load_soundscape_clips(a.soundscapes)
        source = f"{len(a.soundscapes)} soundscape file(s)"
    else:
        clips = load_clips(a.splits, a.csv_dir)
        source = ", ".join(a.splits)
    items = list(clips.items())[: a.limit] if a.limit else list(clips.items())
    print(f"{len(items)} unique clips from {source} -> {a.out}")

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    log_path = os.path.join(a.out, "..", "download_log.csv")
    with open(log_path, "w", newline="") as logf, ThreadPoolExecutor(a.workers) as ex:
        log = csv.writer(logf)
        log.writerow(["youtube_id", "start_time", "status", "note"])
        futs = {ex.submit(download_one, y, s, a.out, a.cookies): (y, s) for y, s in items}
        for i, f in enumerate(as_completed(futs), 1):
            ytid, status, note = f.result()
            counts[status] += 1
            log.writerow([ytid, futs[f][1], status, note])
            if i % 100 == 0 or i == len(items):
                logf.flush()
                print(f"[{i}/{len(items)}] ok={counts['ok']} skipped={counts['skipped']} "
                      f"failed={counts['failed']}", flush=True)

    print(f"Done. Log: {os.path.abspath(log_path)}")
    print("Failed ytids (usually removed/private videos) can be retried by rerunning.")


if __name__ == "__main__":
    main()
