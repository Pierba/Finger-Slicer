"""
Finger Slicer Launcher GUI
==========================
Tkinter front-end that collects the user's choices and shells out to the
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
        ->  Play Game         ->  launches gameplay.py in a new console
"""
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import Tk, ttk, StringVar, BooleanVar, filedialog, messagebox

ROOT_DIR        = Path(__file__).parent
# The entry-point scripts live inside the `src` package and use root-relative
# imports (e.g. `from config import *`, `from src.gameplay.gameplay_utils import
# *`). They must therefore be launched as modules (`python -m ...`) with the
# project root as the working directory, not run directly by file path.
SEGMENT_MODULE  = "src.segmentation.segment_objects"
GAMEPLAY_MODULE = "src.gameplay.gameplay"

# CREATE_NEW_CONSOLE is Windows-only; fall back to 0 on other platforms so the
# script still runs (the subprocess will just inherit the current console).
NEW_CONSOLE_FLAG = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


class LauncherApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Finger Slicer Launcher")
        root.geometry("540x420")
        root.resizable(False, False)

        # User selections
        self.image_path  = StringVar(value="")
        self.model_type  = StringVar(value="yolo")
        self.interactive = BooleanVar(value=False)

        # Handle of the running segmentation process (used to re-enable the
        # Play button once it exits).
        self._segment_proc: subprocess.Popen | None = None

        self.container = ttk.Frame(root, padding=16)
        self.container.pack(fill="both", expand=True)
        self._show_home()

    # =========================================================================
    # screen helpers
    # =========================================================================

    def _clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    # =========================================================================
    # HOME
    # =========================================================================

    def _show_home(self):
        self._clear()
        ttk.Label(self.container, text="Finger Slicer",
                  font=("Segoe UI", 22, "bold")).pack(pady=(30, 6))
        ttk.Label(self.container, text="Choose a program to launch",
                  font=("Segoe UI", 11)).pack(pady=(0, 32))

        ttk.Button(self.container, text="Segment Objects", width=30,
                   command=self._show_segment).pack(pady=8)
        ttk.Button(self.container, text="Play Game", width=30,
                   command=self._run_gameplay).pack(pady=8)

    # =========================================================================
    # SEGMENTATION CONFIG
    # =========================================================================

    def _show_segment(self):
        self._clear()
        ttk.Label(self.container, text="Segment Objects",
                  font=("Segoe UI", 16, "bold")).pack(pady=(4, 14))

        # Actions anchored to the bottom so the rest of the form stacks above.
        action = ttk.Frame(self.container)
        action.pack(side="bottom", fill="x", pady=(16, 0))
        ttk.Button(action, text="Back", command=self._show_home).pack(side="left")
        ttk.Button(action, text="Start Segmentation",
                   command=self._run_segmentation).pack(side="right")

        # --- image picker -----------------------------------------------------
        img_frame = ttk.LabelFrame(self.container, text="Image", padding=10)
        img_frame.pack(fill="x", pady=6)
        ttk.Entry(img_frame, textvariable=self.image_path).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(img_frame, text="Browse...",
                   command=self._pick_image).pack(side="left")

        # --- model selection --------------------------------------------------
        model_frame = ttk.LabelFrame(self.container, text="Model", padding=10)
        model_frame.pack(fill="x", pady=6)
        ttk.Radiobutton(model_frame, text="YOLO  (auto instance segmentation)",
                        value="yolo", variable=self.model_type,
                        command=self._update_modes).pack(anchor="w")
        ttk.Radiobutton(model_frame, text="SAM",
                        value="sam", variable=self.model_type,
                        command=self._update_modes).pack(anchor="w")

        # --- SAM sub-options (shown only when SAM is selected) ----------------
        self.sam_frame = ttk.LabelFrame(self.container, text="SAM mode", padding=10)
        self.sam_mode = StringVar(value="auto" if not self.interactive.get() else "interactive")
        ttk.Radiobutton(self.sam_frame, text="Auto  (fully automatic)",
                        value="auto", variable=self.sam_mode,
                        command=self._sync_interactive).pack(anchor="w")
        ttk.Radiobutton(self.sam_frame, text="Interactive  (click to segment)",
                        value="interactive", variable=self.sam_mode,
                        command=self._sync_interactive).pack(anchor="w")
        self._update_modes()

    def _update_modes(self):
        """Show or hide the SAM sub-options depending on the chosen model."""
        if self.model_type.get() == "sam":
            self.sam_frame.pack(fill="x", pady=6)
        else:
            self.sam_frame.pack_forget()

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

        cmd = [sys.executable, "-m", SEGMENT_MODULE, img,
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
        ttk.Label(self.container, text="Segmentation Running",
                  font=("Segoe UI", 16, "bold")).pack(pady=(24, 10))
        ttk.Label(self.container,
                  text="The segmentation script is running in a separate console.\n"
                       "Close it when you're done — the Play Game button will\n"
                       "light up as soon as it exits.",
                  justify="center").pack(pady=(0, 18))

        self.progress = ttk.Progressbar(self.container, mode="indeterminate")
        self.progress.pack(fill="x", pady=8)
        self.progress.start(12)

        self.status_lbl = ttk.Label(self.container,
                                    text="Waiting for segmentation to finish...")
        self.status_lbl.pack(pady=(10, 6))

        self.play_btn = ttk.Button(self.container, text="Play Game",
                                   command=self._run_gameplay, state="disabled")
        self.play_btn.pack(pady=(10, 4))
        ttk.Button(self.container, text="Back to menu",
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
        self.progress.configure(mode="determinate", value=100)
        self.status_lbl.configure(text="Segmentation finished — ready to play.")
        self.play_btn.configure(state="normal")

    # =========================================================================
    # RUN GAMEPLAY
    # =========================================================================

    def _run_gameplay(self):
        try:
            subprocess.Popen(
                [sys.executable, "-m", GAMEPLAY_MODULE],
                creationflags=NEW_CONSOLE_FLAG, cwd=str(ROOT_DIR))
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))


def main():
    root = Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
