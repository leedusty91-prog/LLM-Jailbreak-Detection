import tkinter as tk
from tkinter import messagebox
import pandas as pd

csv_file = 'data/jailbreak_prompts_DAN.csv'
df = pd.read_csv(csv_file)
df["Labels"] = df["Labels"].fillna(" ")  

labels = [
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
    "Complexity"
]

current_index = df[df['Labels'].str.strip() == ''].index.min()
if pd.isna(current_index): 
    current_index = -1

def submit_labels():
    global current_index
    if current_index == -1:
        messagebox.showinfo("Done")
        save_csv()
        root.quit()
        return
    
    selected_labels = [label for i, label in enumerate(labels) if check_vars[i].get()]
    df.at[current_index, 'Labels'] = ', '.join(selected_labels)
    next_index = df[df['Labels'].str.strip() == ''].index.min()
    if not pd.isna(next_index):  
        current_index = int(next_index)
        show_prompt()
    else:
        current_index = -1
        messagebox.showinfo("Done")
        save_csv() 
        root.quit()

def save_csv():
    df.to_csv(csv_file, index=False)
    messagebox.showinfo("Saved")

def show_prompt():
    if current_index == -1:
        prompt_text.set("No more prompts to label.")
        for var in check_vars:
            var.set(False)
    else:
        prompt_text.set(df.at[current_index, 'prompt'])
        for var in check_vars:
            var.set(False)
        existing_labels = df.at[current_index, 'Labels']
        for i, label in enumerate(labels):
            if label in existing_labels:
                check_vars[i].set(True)

root = tk.Tk()
root.title("Prompt Labeling")
root.attributes('-fullscreen', True)
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

prompt_label = tk.Label(prompt_inner_frame, textvariable=prompt_text, wraplength=1000, font=("Helvetica", font_size), fg="black", bg="white")
prompt_label.pack(fill=tk.BOTH)

def on_prompt_configure(event):
    prompt_canvas.configure(scrollregion=prompt_canvas.bbox("all"))

prompt_inner_frame.bind("<Configure>", on_prompt_configure)

check_vars = [tk.BooleanVar() for _ in labels]

button_frame = tk.Frame(root)
button_frame.pack(pady=20, padx=20, fill=tk.BOTH)

row1_frame = tk.Frame(button_frame)
row2_frame = tk.Frame(button_frame)

row1_frame.pack(pady=10, padx=20)
row2_frame.pack(pady=10, padx=20)

half_len = len(labels) // 2
for i, label in enumerate(labels[:half_len]):
    check_button = tk.Checkbutton(row1_frame, text=label, variable=check_vars[i], font=("Helvetica", font_size))
    check_button.pack(side=tk.LEFT, padx=10, pady=5)

for i, label in enumerate(labels[half_len:], start=half_len):
    check_button = tk.Checkbutton(row2_frame, text=label, variable=check_vars[i], font=("Helvetica", font_size))
    check_button.pack(side=tk.LEFT, padx=10, pady=5)

submit_button = tk.Button(button_frame, text="Next Prompt", command=submit_labels, font=("Helvetica", font_size))
submit_button.pack(pady=10)

save_button = tk.Button(button_frame, text="Save CSV", command=save_csv, font=("Helvetica", font_size))
save_button.pack(pady=10)

if current_index != -1:
    show_prompt()
else:
    prompt_text.set("No prompts to label.")

root.mainloop()
