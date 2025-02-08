import requests
import json

url = "https://jsonplaceholder.typicode.com/todos"
data = requests.get(url).json()

with open("/Users/junsu/Desktop/개인 프로젝트/3rd_assignment/file1.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

with open("/Users/junsu/Desktop/개인 프로젝트/3rd_assignment/file1.json", "r", encoding="utf-8") as f:
    file1 = json.load(f)

print("각 항목의 title 값:")
print("\n------------------------------------------\n")
for item in file1:
    print(item.get("title"))