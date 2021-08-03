import pandas as pd
import os
import inflection

# paths
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Ecommerce.csv')
FT_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data')

df = pd.read_csv(DATA_DIR, encoding = "ISO-8859-1")

# drop unidentified column
df.drop(columns = 'Unnamed: 8', inplace = True)

# data types and df shape
df.dtypes
df.shape

# check na
df.isna().sum()

# droping nas since there's no useful way to fill them
df = df.dropna()
df.isna().sum()

# data types
df.dtypes
df.shape

## data cleaning

# turning columns from camel case to snake case

snakecase = lambda x:inflection.underscore(x)
cols_new = list(map(snakecase, df.columns))
df.columns = cols_new

# changing dtypes
df['invoice_date'] = pd.to_datetime(df.invoice_date)
df['customer_id'] = df.customer_id.astype(int)

# descriptive statistics
## check for the min and max values
df.describe()

len(df[df.quantity < 0])
df = df[df.quantity > 0]
df = df[df.unit_price > 0]

df.head()

# distinct client df
df_cli = df[['customer_id']].drop_duplicates(ignore_index = True)


### feature eng

# basket_size
mix = df.groupby('customer_id').stock_code.nunique().reset_index()
mix.columns = ['customer_id', 'mix']

# revenue
df['gross_revenue'] = df.unit_price * df.quantity
gross_revenue = df.groupby('customer_id').sum()['gross_revenue'].reset_index()

# last_purchase (days)
l_purchase = df.groupby('customer_id').max()['invoice_date'].reset_index() 
l_purchase['last_purchase'] = (df.invoice_date.max() - l_purchase.invoice_date).dt.days

# frequency
frequency = df.groupby('customer_id').invoice_no.nunique().reset_index()
frequency.columns = ['customer_id', 'frequency']

# merging into client df
df_cli = pd.merge(df_cli, gross_revenue, on = 'customer_id', how = 'left')
df_cli = pd.merge(df_cli, mix, on = 'customer_id', how = 'left')
df_cli['last_purchase'] = pd.merge(df_cli, l_purchase, on = 'customer_id', how = 'left')['last_purchase']
df_cli = pd.merge(df_cli, frequency, on = 'customer_id', how = 'left')
df_cli['average_ticket'] = df_cli.gross_revenue/df_cli.frequency

# saving df
df_cli.to_csv(os.path.join(FT_DIR, 'ft_df.csv'), index = False)