# Data Collecting

import pandas as pd

df = pd.read_csv('application_train.csv')

bureau = pd.read_csv('bureau.csv')

df['OCCUPATION_TYPE'].value_counts()

w = ['SK_ID_CURR','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','OWN_CAR_AGE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE','OCCUPATION_TYPE','TARGET']

df1 = df[w]
df1

bureau = pd.read_csv('bureau.csv')
bureau

s = ['SK_ID_CURR', 'SK_ID_BUREAU','AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY', 'CREDIT_DAY_OVERDUE', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG']

bureau1 = bureau[s]
bureau1

data = bureau1.merge(df1, on = 'SK_ID_CURR', how = 'inner')
data

data.info()

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        #Column names
#         Columns = pd.Series(df_train.columns)
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1,ignore_index= True)
        
        # Rename the columns
        mis_val_table = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table = mis_val_table[
            mis_val_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("The dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table

mis_val_table = missing_values_table(data)
mis_val_table

mis_val_table.tail(20)

data['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE'].fillna(data['OCCUPATION_TYPE'].mode()[0])

mis_val_table

data.info()

mis_val_table = missing_values_table(data)
mis_val_table

data.head()

data['AMT_CREDIT_SUM_LIMIT'].value_counts()

# Data Cleaning

data = data.dropna()
data

data.info()

# Exploratory Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure()  
ax = sns.countplot(data = data, x='FLAG_OWN_CAR', hue = 'TARGET')
plt.show() 

fig = plt.figure()  
ax = sns.countplot(data = data, x='FLAG_OWN_REALTY', hue = 'TARGET')
plt.show() 

fig = plt.figure()  
ax = sns.countplot(data = data, y='NAME_INCOME_TYPE', hue = 'TARGET')
plt.show() 

fig = plt.figure()  
ax = sns.countplot(data = data, y='NAME_EDUCATION_TYPE', hue = 'TARGET')
plt.show() 

fig = plt.figure()  
ax = sns.countplot(data = data, y='NAME_HOUSING_TYPE', hue = 'TARGET')
plt.show() 

fig = plt.figure()  
ax = sns.countplot(data = data, y='OCCUPATION_TYPE', hue = 'TARGET')
plt.show() 

fig = plt.figure()  
ax = sns.scatterplot(data = data, x='AMT_INCOME_TOTAL', y = 'AMT_CREDIT_SUM', hue = 'TARGET')
plt.show() 

data['CNT_CREDIT_PROLONG'].value_counts()

fig = plt.figure()  
ax = sns.scatterplot(data = data, x='AMT_INCOME_TOTAL', y = 'CNT_CREDIT_PROLONG', hue = 'TARGET')
plt.show() 

fig = plt.figure()  
ax = sns.scatterplot(data = data, y='NAME_INCOME_TYPE', x = 'CREDIT_DAY_OVERDUE', hue = 'TARGET')
plt.show() 

fig = plt.figure()  
ax = sns.scatterplot(data = data, y='NAME_INCOME_TYPE', x = 'CNT_CREDIT_PROLONG', hue = 'TARGET')
plt.show()

fig = plt.figure()  
ax = sns.barplot(data = data, y='NAME_INCOME_TYPE', x = 'CNT_CREDIT_PROLONG', hue = 'TARGET')
plt.show()

plt.figure(figsize=(10,5))
data.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].mean().plot(kind='bar',color='green')
plt.title('Rata-Rata Income')
plt.ylabel('Pendapatan')

plt.figure(figsize=(10,5))
data.groupby('NAME_INCOME_TYPE')['AMT_CREDIT_SUM'].mean().plot(kind='bar',color='green')
plt.title('Jumlah Kredit Berdasarkan Income Type')
plt.ylabel('Jumlah Kredit')

plt.figure(figsize=(10,5))
data.groupby('NAME_INCOME_TYPE')['CREDIT_DAY_OVERDUE'].mean().plot(kind='bar',color='green')
plt.title('Rata-Rata Jatuh Tempo (hari)')
plt.ylabel('Hari')

# Modelling 

data = pd.get_dummies(data)

data

from sklearn.utils import resample

#create two different dataframe of majority and minority class 
df_majority = data[(data['TARGET']==0)] 
df_minority = data[(data['TARGET']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 25565, # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
data = pd.concat([df_minority_upsampled, df_majority])

data['TARGET'].value_counts()

y = data['TARGET']
X = data.loc[:,data.columns != 'TARGET']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train, y_train)

# Decision Tree
from sklearn import tree
dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

# Evaluasi

y_lr = lr.predict(X_test)
y_dtree = dtree.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_lr))
print(accuracy_score(y_test, y_dtree))

