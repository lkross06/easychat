import pandas as pd
import json
with open("data.json") as intent:
    data = json.load(intent)
df = pd.DataFrame(data['intents'])
print(df.head())
df_patterns = df[['text', 'intent']]
df_responses = df[['responses', 'intent']]
print(df_patterns.head())