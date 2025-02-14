import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = {
    "Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
    "Price": [100, 120, 130, 125, 140]
}
df_price = pd.DataFrame(data)
df_price["Date"] = pd.to_datetime(df_price["Date"])

plt.figure(figsize=(10, 5))
plt.plot(df_price["Date"], df_price["Price"], marker='o', linestyle='-')

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Price by Date")
plt.grid(True)
plt.show()