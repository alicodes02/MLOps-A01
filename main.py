import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
 
df = pd.read_csv('Housing.csv')

df.shape

df.describe()

  
def map_yes_or_no(df,columns):
    for col in columns:
        df[col]=df[col].map({'yes':1,'no':0})
    return df
columns_to_map = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']


df = map_yes_or_no(df, columns_to_map)

  
df.isnull().sum()

  
df.nunique()

  
df.dtypes

  
df

  
NOC = df["furnishingstatus"].unique()
print("Number of Category in within This Feature")

print(len(NOC))

  
df = pd.get_dummies(df,columns=["furnishingstatus"])
df

  
df.rename(columns=lambda x: x.replace('furnishingstatus_', ''), inplace=True)
df.head()

  
df[['furnished', 'semi-furnished', 'unfurnished']] = df[['furnished', 'semi-furnished', 'unfurnished']].astype(int)
df.head()

  
correlation_matrix = df.corr()
plt.figure(figsize=(15, 15))
# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True,cmap='Blues',linewidths=0.5)
# cmap='coolwarm', fmt=".2f", linewidths=0.5
plt.title('Correlation Heatmap')
plt.show()

  

import matplotlib.ticker as mticker # imports the matplotlib.ticker module and assigns it to the alias 'mticker'

plt.figure(figsize=(10, 6))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('Price vs. Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}')) # Now that the module is imported and aliased as 'mticker' this line will execute without error
plt.show()


  
plt.figure(figsize=(10, 6))
sns.boxplot(x='bathrooms', y='price', data=df)
plt.title('Price vs. Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Price')
ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
plt.show()

  
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)
plt.title('mainroad vs Price')
sns.boxplot(x=df.mainroad, y=df.price)
ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))


  
plt.subplot(1,2,2)

sns.boxplot(x=df.guestroom, y=df.price)
ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
plt.title('guestroom vs Price')
plt.show()

  
df.describe()

  
df.hist(bins=50,figsize=(20,20))

  
skewness = df.skew()
skewness = df.skew()

print("Skewness for each feature:\n", skewness)

  
sns.scatterplot(x=df['area'], y=df['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Price vs Area')
plt.show()

  
X = df.drop('price', axis=1)
y = df['price']

scale = StandardScaler()
X = scale.fit_transform(X)

  
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5032)

print(X.shape, X_train.shape, X_test.shape)

  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

  
from sklearn.metrics import r2_score

linear_model_predict = model.predict(X_test)

Rscore = r2_score(y_test, linear_model_predict)
print(f"R squared error for Linear Regression:  {Rscore:.2f}")


