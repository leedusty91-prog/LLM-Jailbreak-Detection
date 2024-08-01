import numpy as np
import pandas as pd

df = pd.read_csv("data/jailbreak_prompts_DAN.csv")

df['is_labelled'] = np.where(df['Labels']==' ', 0, 1)

print(f"Labelled records {df['is_labelled'].sum()} out of {len(df)}")
