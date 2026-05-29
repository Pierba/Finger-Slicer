"""
Finger Slicer Launcher GUI
==========================
CustomTkinter front-end that collects the user's choices and shells out to the
existing entry-point scripts:

  src/segmentation/segment_objects.py   produce RGBA sprites from a source image
  src/gameplay/gameplay.py              play the Fruit-Ninja-style slicing game

Flow
----
  Home  ->  Segment Objects  ->  pick model (YOLO / SAM)
                              ->  if SAM: auto or interactive
                              ->  browse for image
                              ->  Start Segmentation (new console)
                              ->  on exit: enable "Play Game"
        ->  Play Game         ->  launches gameplay.py in a new console,
                                  then closes the launcher menu
"""
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, StringVar, filedialog, messagebox

import customtkinter as ctk

ROOT_DIR        = Path(__file__).parent
SEGMENT_PATH  = str(Path("src/segmentation/segment_objects.py"))
GAMEPLAY_PATH = str(Path("src/gameplay/gameplay.py"))

# CREATE_NEW_CONSOLE is Windows-only; fall back to 0 on other platforms so the
# script still runs (the subprocess will just inherit the current console).
NEW_CONSOLE_FLAG = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

# --- modern look-and-feel ----------------------------------------------------
ctk.set_appearance_mode("dark")        # dark only, for a clean modern look
ctk.set_default_color_theme("blue")    # built-in accent palette

# A friendly green used for the "ready / go" affordances.
ACCENT_GREEN       = "#2FA572"
ACCENT_GREEN_HOVER = "#26895E"


class LauncherApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        root.title("Finger Slicer Launcher")
        # Height is sized for the tallest screen (Segment + SAM mode card shown)
        # so the "Interactive" option never gets clipped.
        root.geometry("560x630")
        root.minsize(560, 630)
        root.resizable(False, False)

        # Shared fonts (created after the root window exists).
        self.font_title    = ctk.CTkFont(family="Segoe UI", size=32, weight="bold")
        self.font_subtitle = ctk.CTkFont(family="Segoe UI", size=14)
        self.font_heading  = ctk.CTkFont(family="Segoe UI", size=22, weight="bold")
        self.font_section  = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        self.font_body     = ctk.CTkFont(family="Segoe UI", size=13)
        self.font_button   = ctk.CTkFont(family="Segoe UI", size=14, weight="bold")

        # User selections
        self.image_path  = StringVar(value="")
        self.model_type  = StringVar(value="yolo")
        self.interactive = BooleanVar(value=False)

        # Handle of the running segmentation process (used to re-enable the
        # Play button once it exits).
        self._segment_proc: subprocess.Popen | None = None

        self.container = ctk.CTkFrame(root, corner_radius=0, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=28, pady=24)
        self._show_home()

    # =========================================================================
    # screen helpers
    # =========================================================================

    def _clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def _card(self, title: str, pack: bool = True):
        """A rounded panel with a small section header; returns (card, body)."""
        card = ctk.CTkFrame(self.container, corner_radius=14)
        if pack:
            card.pack(fill="x", pady=8)
        ctk.CTkLabel(card, text=title.upper(), font=self.font_section,
                     text_color="gray60").pack(anchor="w", padx=18, pady=(14, 0))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=18, pady=(6, 16))
        return card, body

    # =========================================================================
    # HOME
    # =========================================================================

    def _show_home(self):
        self._clear()

        ctk.CTkLabel(self.container, text="🔪  Finger Slicer",
                     font=self.font_title).pack(pady=(48, 4))
        ctk.CTkLabel(self.container, text="Choose a program to launch",
                     font=self.font_subtitle,
                     text_color="gray65").pack(pady=(0, 40))

        ctk.CTkButton(self.container, text="Segment Objects", width=280, height=52,
                      corner_radius=12, font=self.font_button,
                      command=self._show_segment).pack(pady=10)
        ctk.CTkButton(self.container, text="Play Game", width=280, height=52,
                      corner_radius=12, font=self.font_button,
                      fg_color=ACCENT_GREEN, hover_color=ACCENT_GREEN_HOVER,
                      command=self._run_gameplay).pack(pady=10)

    # =========================================================================
    # SEGMENTATION CONFIG
    # =========================================================================

    def _show_segment(self):
        self._clear()
        ctk.CTkLabel(self.container, text="Segment Objects",
                     font=self.font_heading).pack(anchor="w", pady=(4, 14))

        # Actions anchored to the bottom so the rest of the form stacks above.
        action = ctk.CTkFrame(self.container, fg_color="transparent")
        action.pack(side="bottom", fill="x", pady=(18, 0))
        ctk.CTkButton(action, text="← Back", width=110, height=42, corner_radius=10,
                      font=self.font_button, fg_color="transparent", border_width=2,
                      text_color="gray90",
                      command=self._show_home).pack(side="left")
        ctk.CTkButton(action, text="Start Segmentation", width=180, height=42,
                      corner_radius=10, font=self.font_button,
                      fg_color=ACCENT_GREEN, hover_color=ACCENT_GREEN_HOVER,
                      command=self._run_segmentation).pack(side="right")

        # --- image picker -----------------------------------------------------
        _, img_body = self._card("Image")
        ctk.CTkEntry(img_body, textvariable=self.image_path, height=38,
                     corner_radius=8, placeholder_text="No image selected").pack(
                         side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(img_body, text="Browse…", width=96, height=38, corner_radius=8,
                      font=self.font_button, command=self._pick_image).pack(side="left")

        # --- model selection --------------------------------------------------
        _, model_body = self._card("Model")
        ctk.CTkRadioButton(model_body, text="YOLO  (auto instance segmentation)",
                           value="yolo", variable=self.model_type, font=self.font_body,
                           command=self._update_modes).pack(anchor="w", pady=4)
        ctk.CTkRadioButton(model_body, text="SAM", value="sam",
                           variable=self.model_type, font=self.font_body,
                           command=self._update_modes).pack(anchor="w", pady=4)

        # --- SAM sub-options (shown only when SAM is selected) ----------------
        self.sam_card, sam_body = self._card("SAM mode", pack=False)
        self.sam_mode = StringVar(
            value="auto" if not self.interactive.get() else "interactive")
        ctk.CTkRadioButton(sam_body, text="Auto  (fully automatic)", value="auto",
                           variable=self.sam_mode, font=self.font_body,
                           command=self._sync_interactive).pack(anchor="w", pady=4)
        ctk.CTkRadioButton(sam_body, text="Interactive  (click to segment)",
                           value="interactive", variable=self.sam_mode,
                           font=self.font_body,
                           command=self._sync_interactive).pack(anchor="w", pady=4)
        self._update_modes()

    def _update_modes(self):
        """Show or hide the SAM sub-options depending on the chosen model."""
        if self.model_type.get() == "sam":
            self.sam_card.pack(fill="x", pady=8)
        else:
            self.sam_card.pack_forget()

    def _sync_interactive(self):
        self.interactive.set(self.sam_mode.get() == "interactive")

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files",   "*.*"),
            ],
        )
        if path:
            self.image_path.set(path)

    # =========================================================================
    # RUN SEGMENTATION
    # =========================================================================

    def _run_segmentation(self):
        img = self.image_path.get().strip()
        if not img:
            messagebox.showwarning("Missing image", "Please choose an image first.")
            return
        if not Path(img).is_file():
            messagebox.showerror("Invalid image", f"File not found:\n{img}")
            return

        cmd = [sys.executable, SEGMENT_PATH, img,
               "--model-type", self.model_type.get()]
        if self.model_type.get() == "sam" and self.interactive.get():
            cmd.append("-i")

        try:
            self._segment_proc = subprocess.Popen(
                cmd, creationflags=NEW_CONSOLE_FLAG, cwd=str(ROOT_DIR))
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        self._show_segment_running()
        # Watch the process so we can flip the Play button on once it exits.
        threading.Thread(target=self._wait_for_segment, daemon=True).start()

    def _show_segment_running(self):
        self._clear()
        ctk.CTkLabel(self.container, text="Segmentation Running",
                     font=self.font_heading).pack(pady=(40, 12))
        ctk.CTkLabel(self.container,
                     text="The segmentation script is running in a separate console.\n"
                          "Close it when you're done — the Play Game button will\n"
                          "light up as soon as it exits.",
                     font=self.font_body, justify="center",
                     text_color="gray65").pack(pady=(0, 24))

        self.progress = ctk.CTkProgressBar(self.container, mode="indeterminate",
                                           height=12, corner_radius=6)
        self.progress.pack(fill="x", padx=20, pady=8)
        self.progress.start()

        self.status_lbl = ctk.CTkLabel(self.container, font=self.font_body,
                                       text="Waiting for segmentation to finish…")
        self.status_lbl.pack(pady=(14, 18))

        self.play_btn = ctk.CTkButton(self.container, text="Play Game", width=240,
                                      height=48, corner_radius=12, font=self.font_button,
                                      fg_color=ACCENT_GREEN, hover_color=ACCENT_GREEN_HOVER,
                                      command=self._run_gameplay, state="disabled")
        self.play_btn.pack(pady=(6, 10))
        ctk.CTkButton(self.container, text="Back to menu", width=240, height=40,
                      corner_radius=10, font=self.font_button, fg_color="transparent",
                      border_width=2, text_color="gray90",
                      command=self._show_home).pack()

    def _wait_for_segment(self):
        if self._segment_proc is None:
            return
        self._segment_proc.wait()
        # Hop back to the Tk main thread to touch widgets.
        self.root.after(0, self._segment_finished)

    def _segment_finished(self):
        # The user may have navigated away in the meantime.
        if not hasattr(self, "play_btn") or not self.play_btn.winfo_exists():
            return
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(1.0)
        self.progress.configure(progress_color=ACCENT_GREEN)
        self.status_lbl.configure(text="✓  Segmentation finished — ready to play.",
                                  text_color=ACCENT_GREEN)
        self.play_btn.configure(state="normal")

    # =========================================================================
    # RUN GAMEPLAY
    # =========================================================================

    def _run_gameplay(self):
        try:
            subprocess.Popen(
                [sys.executable, GAMEPLAY_PATH],
                creationflags=NEW_CONSOLE_FLAG, cwd=str(ROOT_DIR))
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        # The game runs in its own console, so the launcher menu has done its
        # job — close it. (Any pending segmentation watcher is a daemon thread
        # and dies with the process; the segmentation console keeps running.)
        self.root.destroy()


def main():
    root = ctk.CTk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
