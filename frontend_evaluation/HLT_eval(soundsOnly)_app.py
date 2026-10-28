import os
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import pygame
from gtts import gTTS
from dotenv import load_dotenv
# Initialize Pygame Mixer
pygame.mixer.init()

# Absolute reference to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load variables from the local .env file
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# Read file names dynamically from environment variables with safe fallbacks
OUTPUT_JSON_NAME = os.getenv("OUTPUT_JSON_NAME")
FALLBACK_JSON_NAME = os.getenv("FALLBACK_JSON_NAME")

# Data paths
primary_json = os.path.join(BASE_DIR, "..", "output", OUTPUT_JSON_NAME)
fallback_json = os.path.join(BASE_DIR, FALLBACK_JSON_NAME)
JSON_PATH = primary_json if os.path.exists(primary_json) else fallback_json
EPUB_PATH = os.path.join(BASE_DIR, "TheHobbit.epub")
AUDIO_DIR = os.path.join(BASE_DIR, "ytid")

os.makedirs(AUDIO_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------------------------
def load_data(epub_path, json_path):
    """
    Reads segment data from the NLP JSON pipeline output.
    Filters the data to ONLY include segments that have sound.
    Defaults directly to using the JSON 'text' field for 1:1 segment alignment.
    """
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Returning empty segments.")
        return []
        
    with open(json_path, 'r', encoding='utf-8') as f:
        nlp_data = json.load(f)
        
    segments_data = nlp_data.get("segments", [])

    processed_segments = []
    for seg in segments_data:
        # Only process and display the segment if it actually has sound
        if seg.get("has_sound"):
            # Default directly to the text extracted by the NLP JSON pipeline
            segment_text = seg.get("text", "").strip()

            processed_segments.append({
                "id": seg.get("id"),
                "char_start": seg.get("char_start", 0),
                "char_end": seg.get("char_end", 0),
                "text": segment_text,
                "has_sound": True,
                "clip": seg.get("clip")
            })

    return processed_segments

# ---------------------------------------------------------------------------
# 2. LOCAL AUDIO HANDLER
# ---------------------------------------------------------------------------
def get_clip_audio_path(clip_info):
    if not clip_info:
        return None
        
    ytid = clip_info.get("ytid") or clip_info.get("audiocaps_id")
    if not ytid:
        return None
        
    expected_path = os.path.join(AUDIO_DIR, f"{ytid}.mp3")
    if os.path.exists(expected_path):
        return expected_path
        
    print(f"Audio not found at: {expected_path}")
    return None

# ---------------------------------------------------------------------------
# 3. GUI & EVALUATION CONTROLLER
# ---------------------------------------------------------------------------
class AudioReaderEvaluator:
    def __init__(self, root, segments, eval_out_file="eval_scores.json"):
        self.root = root
        self.segments = segments
        self.eval_out_file = os.path.join(BASE_DIR, eval_out_file)
        
        self.page_size = 5
        self.current_page = 0
        self.active_segment_idx = 0
        self.is_paused = False
        self.scores = {}

        if os.path.exists(self.eval_out_file):
            try:
                with open(self.eval_out_file, "r") as f:
                    self.scores = json.load(f)
            except Exception:
                self.scores = {}

        self.root.title("E-Reader & Audio Evaluation System")
        self.root.geometry("1100x700")
        
        self.setup_ui()
        if self.segments:
            self.load_page(0)
        else:
            self.text_display.insert(tk.END, "Please place demo_nlp.json and TheHobbit.epub in the directory to load data.")

    def setup_ui(self):
        header = tk.Frame(self.root)
        header.pack(side="top", fill="x", padx=10, pady=5)
        
        self.title_label = tk.Label(header, text="The Hobbit - Evaluation Panel", font=("Helvetica", 14, "bold"))
        self.title_label.pack(side="left")
        
        self.page_info = tk.Label(header, text="", font=("Helvetica", 10, "italic"))
        self.page_info.pack(side="right")

        controls = tk.Frame(self.root)
        controls.pack(side="bottom", fill="x", padx=10, pady=10)

        self.prev_btn = tk.Button(controls, text="<- Prev Page", command=self.prev_page, font=("Helvetica", 10))
        self.prev_btn.pack(side="left", padx=5)

        self.pause_btn = tk.Button(controls, text="Pause ⏸", command=self.toggle_pause, font=("Helvetica", 10, "bold"), width=12)
        self.pause_btn.pack(side="left", padx=20)

        self.play_seg_btn = tk.Button(controls, text="Replay Segment 🔄", command=self.play_active_segment, font=("Helvetica", 10))
        self.play_seg_btn.pack(side="left", padx=5)
        
        self.next_seg_btn = tk.Button(controls, text="Next Segment ->", command=self.next_segment, font=("Helvetica", 10))
        self.next_seg_btn.pack(side="left", padx=5)

        self.next_btn = tk.Button(controls, text="Next Page ->", command=self.next_page, font=("Helvetica", 10))
        self.next_btn.pack(side="right", padx=5)

        main_pane = tk.PanedWindow(self.root, orient="horizontal", bd=2, relief="groove")
        main_pane.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        # Left Frame: Reader
        text_frame = tk.Frame(main_pane)
        main_pane.add(text_frame, width=650)

        self.text_display = tk.Text(text_frame, wrap="word", font=("Georgia", 12), spacing1=6, spacing2=4)
        self.text_display.pack(fill="both", expand=True)
        self.text_display.tag_configure("highlight", background="yellow", foreground="black")

        # Right Frame: Evaluation Panel
        eval_frame = tk.Frame(main_pane)
        main_pane.add(eval_frame, width=350)

        tk.Label(eval_frame, text="Segment Evaluation", font=("Helvetica", 12, "bold")).pack(pady=5)

        # Save Button (Always visible at top of right frame)
        save_btn = tk.Button(eval_frame, text="💾 Save Scores JSON", command=self.save_scores, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
        save_btn.pack(side="top", pady=5, fill="x", padx=10)

        # Scrollable Area for Radio Buttons
        canvas_container = tk.Frame(eval_frame)
        canvas_container.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_container)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        
        self.scores_container = tk.Frame(self.canvas)
        self.scores_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas.create_window((0, 0), window=self.scores_container, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def get_page_segments(self):
        start = self.current_page * self.page_size
        end = start + self.page_size
        return self.segments[start:end]

    def load_page(self, page_num):
        self.current_page = page_num
        page_segs = self.get_page_segments()
        total_pages = (len(self.segments) + self.page_size - 1) // self.page_size

        self.page_info.config(text=f"Page {page_num + 1} of {total_pages}")
        
        self.text_display.delete("1.0", tk.END)
        for seg in page_segs:
            self.text_display.insert(tk.END, seg["text"] + "\n\n")

        for child in self.scores_container.winfo_children():
            child.destroy()

        self.score_vars = {}
        for seg in page_segs:
            seg_id = seg["id"]
            row = tk.Frame(self.scores_container, bd=1, relief="groove")
            row.pack(fill="x", pady=6, padx=2)

            label_txt = f"ID #{seg_id} - {'Sound' if seg['has_sound'] else 'No Sound'}"
            tk.Label(row, text=label_txt, font=("Helvetica", 9, "bold")).pack(side="top", anchor="w", padx=2)

            var = tk.StringVar(value=self.scores.get(str(seg_id), ""))
            
            tk.Radiobutton(row, text="1) No sound needed", variable=var, value="no_sound_needed", font=("Helvetica", 9)).pack(anchor="w")
            tk.Radiobutton(row, text="2) Wrong sound", variable=var, value="wrong_sound", font=("Helvetica", 9)).pack(anchor="w")
            tk.Radiobutton(row, text="3) Perfect!", variable=var, value="perfect", font=("Helvetica", 9)).pack(anchor="w")

            self.score_vars[seg_id] = var

        self.active_segment_idx = self.current_page * self.page_size
        self.play_active_segment()

    def highlight_segment(self, target_seg):
        self.text_display.tag_remove("highlight", "1.0", tk.END)
        content = self.text_display.get("1.0", tk.END)
        
        start_idx = content.find(target_seg["text"])
        if start_idx != -1:
            start_pos = f"1.0 + {start_idx} chars"
            end_pos = f"{start_pos} + {len(target_seg['text'])} chars"
            self.text_display.tag_add("highlight", start_pos, end_pos)
            self.text_display.see(start_pos)

    def play_active_segment(self):
        if self.active_segment_idx >= len(self.segments):
            return

        seg = self.segments[self.active_segment_idx]
        self.highlight_segment(seg)

        self.is_paused = False
        self.pause_btn.config(text="Pause ⏸")

        pygame.mixer.stop()
        pygame.mixer.music.stop()

        def _play_thread():
            try:
                tts = gTTS(text=seg["text"], lang="en")
                tts_path = os.path.join(BASE_DIR, "temp_voice.mp3")
                tts.save(tts_path)

                voice = pygame.mixer.Sound(tts_path)
                voice.play()

                if seg.get("has_sound") and seg.get("clip"):
                    effect_file = get_clip_audio_path(seg["clip"])
                    if effect_file:
                        pygame.mixer.music.load(effect_file)
                        pygame.mixer.music.play()
            except Exception as e:
                print(f"Playback error: {e}")

        threading.Thread(target=_play_thread, daemon=True).start()

    def toggle_pause(self):
        if self.is_paused:
            pygame.mixer.unpause()
            pygame.mixer.music.unpause()
            self.pause_btn.config(text="Pause ⏸")
            self.is_paused = False
        else:
            pygame.mixer.pause()
            pygame.mixer.music.pause()
            self.pause_btn.config(text="Play ▶")
            self.is_paused = True

    def record_page_scores(self):
        for seg_id, var in self.score_vars.items():
            val = var.get()
            if val:
                self.scores[str(seg_id)] = val

    def save_scores(self):
        self.record_page_scores()
        with open(self.eval_out_file, "w") as f:
            json.dump(self.scores, f, indent=2)
        messagebox.showinfo("Saved", f"Evaluations saved to {self.eval_out_file}")

    def next_segment(self):
        self.record_page_scores()
        page_end_idx = (self.current_page * self.page_size) + self.page_size - 1
        
        if self.active_segment_idx < page_end_idx and self.active_segment_idx < len(self.segments) - 1:
            self.active_segment_idx += 1
            self.play_active_segment()
        else:
            self.next_page()

    def next_page(self):
        self.record_page_scores()
        total_pages = (len(self.segments) + self.page_size - 1) // self.page_size
        if self.current_page < total_pages - 1:
            self.load_page(self.current_page + 1)

    def prev_page(self):
        self.record_page_scores()
        if self.current_page > 0:
            self.load_page(self.current_page - 1)

# ---------------------------------------------------------------------------
# 4. APPLICATION ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    nlp_segments = load_data(EPUB_PATH, JSON_PATH)
    app = AudioReaderEvaluator(root, segments=nlp_segments)
    root.mainloop()