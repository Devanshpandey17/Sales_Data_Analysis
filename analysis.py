# Sales Data Analysis Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
df = pd.read_csv("data/superstore.csv", encoding='latin1')
print("Dataset Preview:")
print(df.head())


# DATA CLEANING
print("\nChecking missing values:")
print(df.isnull().sum())

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Remove duplicates
df.drop_duplicates(inplace=True)


# FEATURE ENGINEERING

df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year


# EDA
print("\nTotal Sales:", df['Sales'].sum())

# Sales by Category
category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print("\nSales by Category:\n", category_sales)

# Sales by Region
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
print("\nSales by Region:\n", region_sales)

# Monthly Sales Trend
monthly_sales = df.groupby('Month')['Sales'].sum()

# Top 10 Products
top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10)


# VISUALIZATION

plt.figure()
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.savefig("images/monthly_sales.png")

plt.figure()
category_sales.plot(kind='bar')
plt.title("Sales by Category")
plt.savefig("images/category_sales.png")

plt.figure()
region_sales.plot(kind='bar')
plt.title("Sales by Region")
plt.savefig("images/region_sales.png")

plt.figure()
top_products.plot(kind='barh')
plt.title("Top 10 Products")
plt.savefig("images/top_products.png")

print("\nCharts saved in 'images' folder!")


# CORRELATION HEATMAP

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png")

print("\nAnalysis Completed Successfully!")
