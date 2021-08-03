import pandas as pd
import os
import inflection

#paths
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Ecommerce.csv')

df = pd.read_csv(DATA_DIR, encoding = "ISO-8859-1")

#drop unidentified column
df.drop(columns = 'Unnamed: 8', inplace = True)

#data types and df shape
df.dtypes
df.shape

#check na
df.isna().sum()

#droping nas since there's no useful way to fill them
df = df.dropna()
df.isna().sum()

#data types
df.dtypes
df.shape

##data cleaning

#turning columns from camel case to snake case

snakecase = lambda x:inflection.underscore(x)
cols_new = list(map(snakecase, df.columns))
df.columns = cols_new

#changing dtypes
df['invoice_date'] = pd.to_datetime(df.invoice_date)
df['customer_id'] = df.customer_id.astype(int)

#descriptive statistics
df.describe()

##feature eng
#basket_size
#positivation
#revenue
df['cross_revenue'] = df.unit_price * df.quantity
#last_purchase (days)
#frequency
#average 3 months revenue?
#average ticket

df.head()