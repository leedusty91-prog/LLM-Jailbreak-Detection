import re
import os
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, filedialog

BRACKET_RE = re.compile(r"[(){}\[\]<>]")

BG        = "#0f1117"
SURFACE   = "#1a1d27"
SURFACE2  = "#22263a"
ACCENT    = "#7c6af7"
ACCENT2   = "#a78bfa"
TEXT      = "#e8e6f0"
MUTED     = "#6b6880"
SUCCESS   = "#34d399"
WARN      = "#f59e0b"
DANGER    = "#f87171"
BORDER    = "#2e3250"


def _highlight_brackets(text_widget, content):
    text_widget.tag_remove("bracket", "1.0", tk.END)
    for m in BRACKET_RE.finditer(content):
        start = f"1.0 + {m.start()} chars"
        end   = f"1.0 + {m.end()} chars"
        text_widget.tag_add("bracket", start, end)
 
 
def run_editor(df):

    import pandas as pd
 
    mask         = df["prompts"].astype(str).str.contains(BRACKET_RE)
    flagged_idxs = df.index[mask].tolist()
    total        = len(flagged_idxs)
 
    if total == 0:
        messagebox.showinfo("Prompt Editor", "No prompts with brackets found.")
        return df
 
    state = {
        "pos":     0,         
        "edited":  0,        
        "df":      df.copy(),
    }
 
    root = tk.Tk()
    root.title("Prompt Bracket Editor")
    root.configure(bg=BG)
    root.resizable(True, True)
 
    W, H = 820, 560
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
    root.minsize(640, 420)
 
    try:
        mono  = tkfont.Font(family="JetBrains Mono", size=12)
        ui    = tkfont.Font(family="JetBrains Mono", size=10)
        small = tkfont.Font(family="JetBrains Mono", size=9)
        title_f = tkfont.Font(family="JetBrains Mono", size=11, weight="bold")
    except Exception:
        mono  = tkfont.Font(family="Courier", size=12)
        ui    = tkfont.Font(family="Courier", size=10)
        small = tkfont.Font(family="Courier", size=9)
        title_f = tkfont.Font(family="Courier", size=11, weight="bold")
 
    def make_btn(parent, text, cmd, color=ACCENT, fg=TEXT, width=14):
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg=fg, activebackground=ACCENT2, activeforeground=TEXT,
            relief="flat", bd=0, padx=18, pady=10,
            font=ui, cursor="hand2", width=width,
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT2))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn
 
    top = tk.Frame(root, bg=SURFACE, pady=12, padx=20)
    top.pack(fill="x")
 
    counter_var = tk.StringVar()
    counter_lbl = tk.Label(top, textvariable=counter_var, bg=SURFACE,
                           fg=ACCENT2, font=title_f, anchor="w")
    counter_lbl.pack(side="left")
 
    idx_var = tk.StringVar()
    idx_lbl = tk.Label(top, textvariable=idx_var, bg=SURFACE,
                       fg=MUTED, font=small, anchor="e")
    idx_lbl.pack(side="right")
 
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
    path_bar = tk.Frame(root, bg=SURFACE2, pady=8, padx=20)
    path_bar.pack(fill="x")
 
    tk.Label(path_bar, text="Save to:", bg=SURFACE2, fg=MUTED,
             font=small).pack(side="left", padx=(0, 8))
 
    save_path_var = tk.StringVar(value=os.path.join(os.getcwd(), "prompts_edited.csv"))
    path_entry = tk.Entry(path_bar, textvariable=save_path_var, bg=SURFACE,
                          fg=TEXT, insertbackground=ACCENT2, relief="flat",
                          font=small, bd=0)
    path_entry.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 8))
 
    def browse_path():
        chosen = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="prompts_edited.csv",
        )
        if chosen:
            save_path_var.set(chosen)
 
    browse_btn = tk.Button(path_bar, text="Browse", command=browse_path,
                           bg=SURFACE, fg=ACCENT2, activebackground=ACCENT,
                           activeforeground=TEXT, relief="flat", bd=0,
                           font=small, cursor="hand2", padx=10, pady=4)
    browse_btn.pack(side="left")
 
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
 
    info_var = tk.StringVar()
    info_lbl = tk.Label(root, textvariable=info_var, bg=BG, fg=WARN,
                        font=small, anchor="w", padx=24, pady=8)
    info_lbl.pack(fill="x")
 
    editor_frame = tk.Frame(root, bg=BG, padx=20)
    editor_frame.pack(fill="both", expand=True)
 
    scrollbar = tk.Scrollbar(editor_frame, bg=SURFACE2,
                              troughcolor=SURFACE, relief="flat", bd=0)
    scrollbar.pack(side="right", fill="y", pady=4)
 
    editor = tk.Text(
        editor_frame,
        font=mono, bg=SURFACE, fg=TEXT,
        insertbackground=ACCENT2,
        selectbackground=ACCENT, selectforeground=TEXT,
        relief="flat", bd=0,
        wrap="word", padx=14, pady=12,
        yscrollcommand=scrollbar.set,
        spacing1=4, spacing3=4,
    )
    editor.pack(fill="both", expand=True, pady=4)
    scrollbar.config(command=editor.yview)
 
    editor.tag_configure("bracket", foreground=DANGER,
                         background="#3a1a1a", font=mono)
 
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
    bottom = tk.Frame(root, bg=SURFACE, pady=14, padx=20)
    bottom.pack(fill="x")
 
    exit_btn = make_btn(bottom, "✕  Exit & Save", None, color="#2a2d3e",
                        fg=MUTED, width=16)
    exit_btn.pack(side="left")
 
    right = tk.Frame(bottom, bg=SURFACE)
    right.pack(side="right")
 
    skip_btn   = make_btn(right, "Skip →",   None, color=SURFACE2, fg=TEXT, width=10)
    submit_btn = make_btn(right, "✓  Submit", None, color=SUCCESS,  fg=BG,  width=12)
    skip_btn.pack(side="left", padx=(0, 10))
    submit_btn.pack(side="left")
 
    status_var = tk.StringVar(value="")
    status_lbl = tk.Label(root, textvariable=status_var, bg=BG, fg=MUTED,
                          font=small, anchor="w", padx=24, pady=4)
    status_lbl.pack(fill="x")
 
    def refresh_counter():
        pos     = state["pos"]
        edited  = state["edited"]
        counter_var.set(f"Edited  {edited} / {total}")
        if pos < total:
            idx_var.set(f"DataFrame row index: {flagged_idxs[pos]}")
        else:
            idx_var.set("")
 
    def load_prompt(pos):
        if pos >= total:
            finish()
            return
        real_idx = flagged_idxs[pos]
        text     = str(state["df"].at[real_idx, "prompts"])
 
        info_var.set(f"  ⚠  Brackets detected — edit the prompt below, then click Submit.")
        editor.config(state="normal")
        editor.delete("1.0", tk.END)
        editor.insert("1.0", text)
        _highlight_brackets(editor, text)
        editor.focus_set()
        status_var.set("")
        refresh_counter()
        def on_key(event):
            content = editor.get("1.0", tk.END).rstrip("\n")
            _highlight_brackets(editor, content)
        editor.bind("<KeyRelease>", on_key)
 
    def do_submit():
        pos      = state["pos"]
        if pos >= total:
            return
        real_idx = flagged_idxs[pos]
        new_text = editor.get("1.0", tk.END).rstrip("\n")
        state["df"].at[real_idx, "prompts"] = new_text
        state["edited"] += 1
        status_var.set(f"  ✓  Row {real_idx} updated.")
        state["pos"] += 1
        load_prompt(state["pos"])
 
    def do_skip():
        pos = state["pos"]
        if pos >= total:
            return
        real_idx = flagged_idxs[pos]
        status_var.set(f"  –  Row {real_idx} skipped.")
        state["pos"] += 1
        load_prompt(state["pos"])
 
    def save_df():
        path = save_path_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Please enter a save path first.", parent=root)
            return False
        try:
            state["df"].to_csv(path, index=False)
            status_var.set(f"  ✓  Saved to {path}")
            return True
        except Exception as e:
            messagebox.showerror("Save failed", str(e), parent=root)
            return False
 
    def finish():
        info_var.set("  ✓  All flagged prompts reviewed.")
        editor.delete("1.0", tk.END)
        editor.config(state="disabled")
        submit_btn.config(state="disabled")
        skip_btn.config(state="disabled")
        counter_var.set(f"Edited  {state['edited']} / {total}  — Done!")
        idx_var.set("")
        save_df()
 
    def do_exit():
        if messagebox.askyesno("Save & Exit",
                               "Save changes and close the editor?",
                               parent=root):
            save_df()
            root.quit()
 
    submit_btn.config(command=do_submit)
    skip_btn.config(command=do_skip)
    exit_btn.config(command=do_exit)
    root.protocol("WM_DELETE_WINDOW", do_exit)
    root.bind("<Return>",   lambda e: do_submit())
    root.bind("<Escape>",   lambda e: do_skip())

    load_prompt(0)
    root.mainloop()
 
    try:
        root.destroy()
    except Exception:
        pass
 
    return state["df"]



if __name__ == "__main__":
    import pandas as pd


    df = pd.read_csv('./data/jailbreak_prompts_DAN.csv') #PATH TO DF
    df = df.rename(columns={"prompt": "prompts"})
    df = run_editor(df)
    print("Program Exited.")