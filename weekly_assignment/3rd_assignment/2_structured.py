import pandas as pd

data = {
    "ID": [1, 2, 3, 4, 5],
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 32, 45, 29, 40],
    "Department": ["HR", "Finance", "IT", "Marketing", "IT"],
    "Salary": [48000, 52000, 60000, 45000, 70000]
}
df = pd.DataFrame(data)

filtered_df = df.query("Age >= 30 and Salary >= 50000")
result = filtered_df[["Name", "Age", "Department"]]
print(result)