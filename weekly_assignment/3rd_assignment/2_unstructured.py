import pandas as pd
import requests

response = requests.get("https://jsonplaceholder.typicode.com/users")
data = response.json()
df = pd.DataFrame(data)

df['City'] = df['address'].apply(lambda addr: addr['city'])
df['Company'] = df['company'].apply(lambda comp: comp['name'])
df_new = df[['id', 'name', 'username', 'email', 'City', 'Company']].rename(
    columns={
        'id': 'ID',
        'name': 'Name',
        'username': 'Username',
        'email': 'Email'
    }
)

filtered_df = df_new[df_new['City'].isin(["Lebsackbury", "Roscoeview"])]
filtered_df.to_csv("filtered_users.csv", index=False)
print(filtered_df)