import pandas as pd
import numpy as np


df1 = pd.read_csv('data/jailbreak_prompts_78.csv')
df2 = pd.read_csv('data/jailbreak_prompts_DAN.csv')
df3 = pd.read_csv('data/regular_prompts_DAN.csv')
df4 = pd.read_csv('data/harmbench_behaviors_text_all.csv')

#df2['jailbreak'] = df2['jailbreak'].replace({True: 1, False: 0})
#df3['jailbreak'] = df3['jailbreak'].replace({True: 1, False: 0})

print(df4['SemanticCategory'].unique())
