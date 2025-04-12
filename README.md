<H3>ENTER YOUR NAME</H3> PRIYAADARSHINI.K
<H3>ENTER YOUR REGISTER NO.</H3> 212223240126
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
TYPE YOUR CODE HERE
```
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# Read the dataset
df = pd.read_csv('Churn_Modelling.csv')
print("df.head()")
df.head()
print("df.tail()")
df.tail()
df.info()
print(df)
# Split the dataset
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
print(x)
print(y)
# Find the missing values 
print(df.isnull().sum())
# Handling Missing values 
print(df.isnull().sum())
y = df.iloc[:,-1].values
print(y)
# Check for the duplicates
df.duplicated()
# Detect outliners
des=df['CreditScore'].describe()
print(des)
# Normalize the dataset
df_numeric = df.select_dtypes(include=[np.number])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
# Splitting the data for training & testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)  
print(len(x_train))
print(x_test)  
print(len(x_test))

```


## OUTPUT:
SHOW YOUR OUTPUT HERE

df.head()
![alt text](image-2.png)

df.tail()
![alt text](image-3.png)

Missing values

![alt text](image-4.png)

Training values of x and y
![alt text](image-5.png)

Testing values of x and y
![alt text](image-6.png)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


