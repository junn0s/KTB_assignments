import pandas as pd

data = {
    "Year": [2023, 2023, 2023],
    "Quarter": ["Q1", "Q2", "Q3"],
    "Sales": [200, 300, 250]
}
df = pd.DataFrame(data)

df["Total_Sales"] = df.groupby("Year")["Sales"].transform("sum")
print(df)