# dataset source --> kaggle
# link to source --> https://www.kaggle.com/datasets/ak0212/indian-car-market-dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/minor project 1/my_dataset/car_dataset_india.csv")
print("Dataset shape :", df.shape) # it will show the shape of our dataset

print("                             <--------------------------------------------------------------------------->")

print("Column Names :", df.columns.to_list())

print("                             <--------------------------------------------------------------------------->")
print("Data types :", df.dtypes)

print("                             <--------------------------------------------------------------------------->")
rows_before_cleaning = len(df)
# seperate measurements, identifiers, and categories
categorical_cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
numeric_cols = [col for col in df.select_dtypes(include='number').columns if col != 'Car_ID']
print("Numeric measurement colums :", numeric_cols)
print("                             <--------------------------------------------------------------------------->")
print("Categorical columns :", categorical_cols)
print("                             <--------------------------------------------------------------------------->")
print("Identifier column : Car_ID")
print("                             <--------------------------------------------------------------------------->")

# printing the first 5 and last five rows
print(df.head(), '\n')
print("                             <--------------------------------------------------------------------------->")

print("\n", df.tail(), "\n")

print("                             <--------------------------------------------------------------------------->")

# printing the data types
print("\n", df.info(), "\n")

print("                             <--------------------------------------------------------------------------->")

# printing the description of records(like mean, std, max, etc...)
print("\n", df.describe(), "\n")

print("                             <--------------------------------------------------------------------------->")

# now we will check whether there is some missing values or not in the data
print("\n", df.isnull().sum(), "\n")

print("                             <--------------------------------------------------------------------------->")

# now we will check if there is any duplicated value in our data
print("\n", df.duplicated().sum(), "\n")


# our dataset does'nt contains any null or duplicated values, if it was there we can perform following commmands :
# df.drop_duplicates(inplace = True) # for removing duplicated values
# df.fillna(df.mode().iloc[0]) #replacing with mode for objective data type
# df.fillna(df.median())

# print("                             <--------------------------------------------------------------------------->")

# now outliers

# here we can find whether there are some outliers or not
df.select_dtypes(include='number').plot(kind='box', figsize=(7, 7), rot = 45)
# plt.show()

print("                             <--------------------------------------------------------------------------->")

# now we will check which aspect is most common in our data and its occurences
most_common_aspect = df['Brand'].value_counts().idxmax()
highest_count = df['Brand'].value_counts().max()
print(f"The most common aspect of Brand is '{most_common_aspect}' with '{highest_count}' occurrences\n")
least_common_aspect = df['Brand'].value_counts().idxmin()
lowest_count = df['Brand'].value_counts().min()
print(f"The least common aspect is '{least_common_aspect} with '{lowest_count}' occurrences\n")

print("                             <--------------------------------------------------------------------------->")

# now for fuel types we will see
most_common_fuel = df['Fuel_Type'].value_counts().idxmax()
highest_count1 = df['Fuel_Type'].value_counts().max()
print(f"The most common aspect of Fuel Types is '{most_common_aspect}' with '{highest_count1}' occurrences\n")
least_common_fuel = df['Fuel_Type'].value_counts().idxmin()
lowest_count1 = df['Fuel_Type'].value_counts().min()
print(f"The least common aspect of Fuel Types is '{least_common_aspect}' with '{lowest_count1}' occurrences\n")

print("                             <--------------------------------------------------------------------------->")

# less check how car prices are distributed
print(df['Price'].describe())

print("                             <--------------------------------------------------------------------------->")

# less check whether mileage is influencing price of cars
print(df[['Mileage', 'Price']].corr())

print("                             <--------------------------------------------------------------------------->")
# now we will check for graphs ==>
# 1. Price distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Price", bins=30, kde=True, color="#2C8F8F")
plt.axvline(df["Price"].median(), color="darkorange", linestyle="--",
            label=f"Median: {df['Price'].median():,.0f}")
plt.title("Car Price Distribution")
plt.xlabel("Price (CSV units)")
plt.ylabel("Number of records")
plt.legend()
plt.tight_layout()
plt.show()


# 2. Brand counts (a clearer bar chart than the Brand histogram in your file)
plt.figure(figsize=(10, 5))
brand_order = df["Brand"].value_counts().index
sns.countplot(data=df, x="Brand", order=brand_order, color="#5B86B4")
plt.title("Number of Records by Brand")
plt.xlabel("Brand")
plt.ylabel("Number of records")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.show()


# 3. Fuel type counts
plt.figure(figsize=(8, 5))
fuel_order = df["Fuel_Type"].value_counts().index
sns.countplot(data=df, x="Fuel_Type", order=fuel_order, color="#2D8585")
plt.title("Number of Records by Fuel Type")
plt.xlabel("Fuel type")
plt.ylabel("Number of records")
plt.tight_layout()
plt.show()


# 4. Price by transmission
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Transmission", y="Price",
            order=["Manual", "Automatic"], color="#8FB4AE")
plt.title("Price by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Price (CSV units)")
plt.tight_layout()
plt.show()


# 5. Mileage and price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Mileage", y="Price", hue="Transmission",
                alpha=0.45, s=20)
plt.title("Mileage and Price")
plt.xlabel("Mileage (CSV units)")
plt.ylabel("Price (CSV units)")
plt.tight_layout()
plt.show()


# 6. Model year and price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Year", y="Price", alpha=0.35, s=20, color="#2D8585")
plt.title("Model Year and Price")
plt.xlabel("Year")
plt.ylabel("Price (CSV units)")
plt.tight_layout()
plt.show()


# 7. Average price by brand (a simpler version of your Brand/Price plot)
brand_average = (
    df.groupby("Brand", as_index=False)["Price"]
      .mean()
      .sort_values("Price")
)

plt.figure(figsize=(9, 6))
sns.barplot(data=brand_average, x="Price", y="Brand", color="#2D8585")
plt.title("Average Price by Brand")
plt.xlabel("Average price (CSV units)")
plt.ylabel("Brand")
plt.tight_layout()
plt.show()


# now we will change the string d_type to numeric(here, int)
cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
for col in cols:
    print(col, df[col].unique())
print("\n")

print("                             <--------------------------------------------------------------------------->")

# less convert noww
df = pd.get_dummies(df, columns=cols, dtype=int)

# less print after cleaning the d_type
print("\n", df.info())