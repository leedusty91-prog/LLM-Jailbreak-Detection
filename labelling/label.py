"""GUI for assigning jailbreak-type labels.

Checkbox state is derived from ``labels.parse_labels`` so the three on-disk
formats (list literals, quoted names, and plain comma-separated names) all
restore the same boxes. A label is selected only on an exact name match.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import pandas as pd

from labels import parse_labels, selected_flags
from paths import data_path

CSV_FILE = data_path("jailbreak_prompts_DAN.csv")
LABELS = [
    "Character Roleplay",
    "Assumed Responsibility",
    "Research Experiment",
    "Text Continuation",
    "Logical Reasoning",
    "Program Execution",
    "Translation",
    "Superior Model",
    "Sudo Mode",
    "Simulate Jailbreaking",
    "Ethical Appeal",
    "Contrastive",
    "Contradiction",
    "Complexity",
    "Gameplay",
    "Non-Jailbreak",
]


def load_prompts(csv_file=CSV_FILE) -> pd.DataFrame:
    frame = pd.read_csv(csv_file)
    frame["Labels"] = frame["Labels"].fillna(" ")
    return frame


def first_unlabelled_index(frame: pd.DataFrame):
    unlabelled = frame["Labels"].map(lambda value: len(parse_labels(value)) == 0)
    if not unlabelled.any():
        return -1
    return int(frame.index[unlabelled][0])


def main():
    import tkinter as tk
    from tkinter import messagebox

    csv_file = CSV_FILE
    frame = load_prompts(csv_file)
    current_index = first_unlabelled_index(frame)

    def save_csv():
        frame.to_csv(csv_file, index=False)
        messagebox.showinfo("Saved", f"Saved {csv_file}")

    def show_prompt():
        if current_index == -1:
            prompt_text.set("No more prompts to label.")
            for var in check_vars:
                var.set(False)
            return
        prompt_text.set(frame.at[current_index, "prompt"])
        for var, selected in zip(check_vars, selected_flags(frame.at[current_index, "Labels"], LABELS)):
            var.set(selected)

    def submit_labels():
        nonlocal current_index
        if current_index == -1:
            messagebox.showinfo("Done", "No prompts left to label.")
            save_csv()
            root.quit()
            return
        chosen = [label for label, var in zip(LABELS, check_vars) if var.get()]
        frame.at[current_index, "Labels"] = ", ".join(chosen)
        nxt = first_unlabelled_index(frame)
        if nxt != -1:
            current_index = nxt
            show_prompt()
        else:
            current_index = -1
            messagebox.showinfo("Done", "No prompts left to label.")
            save_csv()
            root.quit()

    def previous_prompt():
        nonlocal current_index
        if current_index < 0:
            return
        chosen = [label for label, var in zip(LABELS, check_vars) if var.get()]
        frame.at[current_index, "Labels"] = ", ".join(chosen)
        if current_index == 0:
            messagebox.showinfo("Start", "You are at the first prompt")
            return
        current_index -= 1
        show_prompt()

    def next_prompt():
        nonlocal current_index
        if current_index < 0:
            return
        chosen = [label for label, var in zip(LABELS, check_vars) if var.get()]
        frame.at[current_index, "Labels"] = ", ".join(chosen)
        if current_index + 1 >= len(frame):
            messagebox.showinfo("End", "Last prompt")
            return
        current_index += 1
        show_prompt()

    root = tk.Tk()
    root.title("Prompt Labeling")
    root.attributes("-fullscreen", True)
    font_size = 18
    prompt_text = tk.StringVar()
    prompt_frame = tk.Frame(root)
    prompt_frame.pack(pady=15, padx=15, fill=tk.BOTH)
    prompt_canvas = tk.Canvas(prompt_frame, height=400)
    prompt_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    prompt_scrollbar = tk.Scrollbar(prompt_frame, orient=tk.VERTICAL, command=prompt_canvas.yview)
    prompt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    prompt_canvas.configure(yscrollcommand=prompt_scrollbar.set)
    prompt_inner_frame = tk.Frame(prompt_canvas)
    prompt_canvas.create_window((0, 0), window=prompt_inner_frame, anchor="nw")
    prompt_label = tk.Label(
        prompt_inner_frame,
        textvariable=prompt_text,
        wraplength=1000,
        font=("Helvetica", font_size),
        fg="black",
        bg="white",
    )
    prompt_label.pack(fill=tk.BOTH)

    def on_prompt_configure(_event):
        prompt_canvas.configure(scrollregion=prompt_canvas.bbox("all"))

    prompt_inner_frame.bind("<Configure>", on_prompt_configure)
    check_vars = [tk.BooleanVar() for _ in LABELS]
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10, padx=20, fill=tk.BOTH)
    button_frame2 = tk.Frame(root)
    button_frame2.pack(pady=10, padx=20, fill=tk.BOTH)
    row_frames = [tk.Frame(button_frame) for _ in range(3)]
    for row in row_frames:
        row.pack(pady=10, padx=20)
    third_len = len(LABELS) // 3
    spans = (
        (0, third_len, row_frames[0]),
        (third_len, len(LABELS) - third_len, row_frames[1]),
        (len(LABELS) - third_len, len(LABELS), row_frames[2]),
    )
    for start, stop, row in spans:
        for index in range(start, stop):
            tk.Checkbutton(
                row, text=LABELS[index], variable=check_vars[index], font=("Helvetica", font_size)
            ).pack(side=tk.LEFT, padx=10, pady=5)
    tk.Button(button_frame, text="Previous Prompt", command=previous_prompt, font=("Helvetica", font_size)).pack(
        side="left", pady=10
    )
    tk.Button(button_frame, text="Next Prompt", command=next_prompt, font=("Helvetica", font_size)).pack(
        side="right", pady=10
    )
    tk.Button(button_frame2, text="Next Unlabelled", command=submit_labels, font=("Helvetica", font_size)).pack(
        side="right", pady=10
    )
    tk.Button(button_frame2, text="Save CSV", command=save_csv, font=("Helvetica", font_size)).pack(
        side="right", pady=10
    )
    if current_index != -1:
        show_prompt()
    else:
        prompt_text.set("No prompts to label.")
    root.mainloop()


if __name__ == "__main__":
    main()
