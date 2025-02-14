import pandas as pd

df = pd.DataFrame({"이름": [" John Doe ", "Alice ", " Bob", "Charlie Doe "]})
df["이름"] = df["이름"].str.strip()
filtered_df = df[df["이름"].str.contains("Doe")]
print(filtered_df)
