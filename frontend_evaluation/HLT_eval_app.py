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

# Initialize Pygame Mixer
pygame.mixer.init()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default to the output folder file
primary_json = os.path.join(BASE_DIR, "..", "output", "hobbit.minilm.gated.soundscape.json")
fallback_json = os.path.join(BASE_DIR, "demo_nlp.json")

JSON_PATH = primary_json if os.path.exists(primary_json) else fallback_json

# ---------------------------------------------------------------------------
# 1. DATA PREPARATION (REAL NLP JSON + EPUB SLICING)
# ---------------------------------------------------------------------------
def load_data(epub_path, json_path):
    """
    Reads the EPUB text and slices it exactly using the char_start and char_end
    indices provided by the NLP JSON file.
    """
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Returning empty segments.")
        return []
        
    with open(json_path, 'r', encoding='utf-8') as f:
        nlp_data = json.load(f)
        
    segments_data = nlp_data.get("segments", [])

    # Extract raw text from EPUB to slice against
    full_text = ""
    if os.path.exists(epub_path):
        book = epub.read_epub(epub_path)
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        for item in items:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            full_text += soup.get_text() + " "
    else:
        print(f"Warning: {epub_path} not found. Relying on JSON 'text' field if available.")

    processed_segments = []
    for seg in segments_data:
        start = seg.get("char_start", 0)
        end = seg.get("char_end", 0)
        
        # Prioritize slicing from the actual EPUB text using the character indexes
        if full_text and end > start:
            segment_text = full_text[start:end].strip()
        else:
            segment_text = seg.get("text", "").strip()

        processed_segments.append({
            "id": seg.get("id"),
            "char_start": start,
            "char_end": end,
            "text": segment_text,
            "has_sound": seg.get("has_sound", False),
            "clip": seg.get("clip")
        })

    return processed_segments

# ---------------------------------------------------------------------------
# 2. LOCAL AUDIO HANDLER (REAL AUDIO DIRECTORY)
# ---------------------------------------------------------------------------
AUDIO_DIR = "./ytid" 
# Ensure the directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

def get_clip_audio_path(clip_info):
    """
    Checks for {ytid}.mp3 locally inside the AUDIO_DIR.
    For the demo, using "sampleAudio.mp3" in the folder 
    and using "ytid": "sampleAudio" in JSON.
    """
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
        self.eval_out_file = eval_out_file
        
        #State Tracking
        self.page_size = 5  # Segments shown per page
        self.current_page = 0
        self.active_segment_idx = 0
        self.is_paused = False
        self.scores = {}  # {segment_id: score_int}

        #Load existing scores if present
        if os.path.exists(self.eval_out_file):
            try:
                with open(self.eval_out_file, "r") as f:
                    self.scores = json.load(f)
            except Exception:
                self.scores = {}

        self.root.title("E-Reader & Audio Evaluation System")
        self.root.geometry("1000x650")
        
        self.setup_ui()
        if self.segments:
            self.load_page(0)
        else:
            self.text_display.insert(tk.END, "Please place demo_nlp.json and TheHobbit.epub in the directory to load data.")

    def setup_ui(self):
        # Header Frame (Top)
        header = tk.Frame(self.root)
        header.pack(side="top", fill="x", padx=10, pady=5)
        
        self.title_label = tk.Label(header, text="The Hobbit - Evaluation Panel", font=("Helvetica", 14, "bold"))
        self.title_label.pack(side="left")
        
        self.page_info = tk.Label(header, text="", font=("Helvetica", 10, "italic"))
        self.page_info.pack(side="right")

        #Bottom Frame: Player Controls (Packed BEFORE main pane so it is never pushed off screen)
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

        # Main Layout: Left = Reader, Right = Evaluation Sidebar
        main_pane = tk.PanedWindow(self.root, orient="horizontal", bd=2, relief="groove")
        main_pane.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        # Left Frame: Text Display
        text_frame = tk.Frame(main_pane)
        main_pane.add(text_frame, width=650)

        self.text_display = tk.Text(text_frame, wrap="word", font=("Georgia", 12), spacing1=6, spacing2=4)
        self.text_display.pack(fill="both", expand=True)
        self.text_display.tag_configure("highlight", background="yellow", foreground="black")

        # Right Frame: Evaluation Panel
        eval_frame = tk.Frame(main_pane)
        main_pane.add(eval_frame, width=300)

        tk.Label(eval_frame, text="Segment Scoring (1-10)", font=("Helvetica", 12, "bold")).pack(pady=10)

        # Scrollable container for segment scores on current page
        self.scores_container = tk.Frame(eval_frame)
        self.scores_container.pack(fill="both", expand=True, padx=5)

        # Save Button
        save_btn = tk.Button(eval_frame, text="💾 Save Scores JSON", command=self.save_scores, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
        save_btn.pack(side="bottom", pady=10, fill="x", padx=10)

    def get_page_segments(self):
        start = self.current_page * self.page_size
        end = start + self.page_size
        return self.segments[start:end]

    def load_page(self, page_num):
        self.current_page = page_num
        page_segs = self.get_page_segments()
        total_pages = (len(self.segments) + self.page_size - 1) // self.page_size

        self.page_info.config(text=f"Page {page_num + 1} of {total_pages}")
        
        #Render Text
        self.text_display.delete("1.0", tk.END)
        for seg in page_segs:
            self.text_display.insert(tk.END, seg["text"] + "\n\n")

        #Build Sidebar Rating Widgets
        for child in self.scores_container.winfo_children():
            child.destroy()

        self.score_entries = {}
        for seg in page_segs:
            seg_id = seg["id"]
            row = tk.Frame(self.scores_container)
            row.pack(fill="x", pady=4)

            label_txt = f"ID #{seg_id} ({'Sound' if seg['has_sound'] else 'No Sound'}):"
            tk.Label(row, text=label_txt, font=("Helvetica", 9), anchor="w").pack(side="left")

            entry = tk.Entry(row, width=5)
            entry.pack(side="right")
            if str(seg_id) in self.scores:
                entry.insert(0, str(self.scores[str(seg_id)]))

            self.score_entries[seg_id] = entry

        #Set active segment to the first one on the page and play
        self.active_segment_idx = self.current_page * self.page_size
        self.play_active_segment()

    def highlight_segment(self, target_seg):
        """Highlights active segment text inside Tkinter Text Widget."""
        self.text_display.tag_remove("highlight", "1.0", tk.END)
        content = self.text_display.get("1.0", tk.END)
        
        start_idx = content.find(target_seg["text"])
        if start_idx != -1:
            start_pos = f"1.0 + {start_idx} chars"
            end_pos = f"{start_pos} + {len(target_seg['text'])} chars"
            self.text_display.tag_add("highlight", start_pos, end_pos)
            self.text_display.see(start_pos)

    def play_active_segment(self):
        """Plays TTS Narration + Audio Effect for active segment."""
        if self.active_segment_idx >= len(self.segments):
            return

        seg = self.segments[self.active_segment_idx]
        self.highlight_segment(seg)

        #Ensure UI resets button state
        self.is_paused = False
        self.pause_btn.config(text="Pause ⏸")

        #Stop ongoing audio
        pygame.mixer.stop()
        pygame.mixer.music.stop()

        #Threaded audio handling to keep UI responsive
        def _play_thread():
            try:
                # 1. Generate & Play TTS
                tts = gTTS(text=seg["text"], lang="en")
                tts_path = os.path.join("./", "temp_voice.mp3")
                tts.save(tts_path)

                voice = pygame.mixer.Sound(tts_path)
                voice.play()

                # 2. Fetch & Play Audio Effect (if present)
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
        """Collects scores from current page entries."""
        for seg_id, entry in self.score_entries.items():
            val = entry.get().strip()
            if val.isdigit():
                self.scores[str(seg_id)] = int(val)

    def save_scores(self):
        self.record_page_scores()
        with open(self.eval_out_file, "w") as f:
            json.dump(self.scores, f, indent=2)
        messagebox.showinfo("Saved", f"Evaluations saved to {self.eval_out_file}")

    def next_segment(self):
        """Advances to the next segment on the page, or flips the page if at the end."""
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
    
    #Load real data using the EPUB and the JSON output from the NLP pipeline
    nlp_segments = load_data("TheHobbit.epub", JSON_PATH)
    
    app = AudioReaderEvaluator(root, segments=nlp_segments)
    root.mainloop()