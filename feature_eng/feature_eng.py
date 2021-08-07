import pandas as pd
import os
import inflection
import numpy as np
import re

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

# turning columns from camel case to snake case
snakecase = lambda x:inflection.underscore(x)
cols_new = list(map(snakecase, df.columns))
df.columns = cols_new

# changing dtypes
df['invoice_date'] = pd.to_datetime(df.invoice_date)
df['customer_id'] = df.customer_id.astype(int)


## descriptive statistics
num_attrs = df.select_dtypes(include = ['int64', 'int32', 'float64'])
cat_attrs = df.select_dtypes(exclude = ['int64', 'int32', 'float64', 'datetime64[ns]'])

# central tendency - mean, median
ct1 = pd.DataFrame(num_attrs.apply(np.mean)).T
ct2 = pd.DataFrame(num_attrs.apply(np.median)).T

# dispersion - std, min, max, range, skew, kurtosis
d1 = pd.DataFrame(num_attrs.apply(np.std)).T
d2 = pd.DataFrame(num_attrs.apply(np.min)).T
d3 = pd.DataFrame(num_attrs.apply(np.max)).T
d4 = pd.DataFrame(num_attrs.apply(lambda x: x.max() - x.min())).T
d5 = pd.DataFrame(num_attrs.apply(lambda x: x.skew())).T
d6 = pd.DataFrame(num_attrs.apply(lambda x: x.kurtosis())).T    

stats = pd.concat( [d2, d3, d4, ct1, ct2, d1, d5, d6]).T.reset_index()
stats.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std','skew', 'kurtosis']
stats



## Filtering variables

# contains "POST"
# df[cat_attrs.stock_code.apply(lambda x: bool(re.search('[^0-9]+', x)) )]

# filtering out stock code containing strings
df = df[~cat_attrs.stock_code.apply(lambda x: bool(re.search('^[a-zA-Z]+$', x)) )]

# removing price equal to 0
df = df[df.unit_price > 0]

# creating revenue and return dataframe
df_pos = df[df.quantity > 0]
df_neg = df[df.quantity < 0]

# removing countries not identified
df2 = df2[~df2['country'].isin( ['European Community', 'Unspecified' ] ) ]



## feature eng

# distinct client df
df_cli = df[['customer_id']].drop_duplicates(ignore_index = True)


# basket_size
mix = df_pos.groupby('customer_id').stock_code.nunique().reset_index()
mix.columns = ['customer_id', 'mix']

# revenue and returned amount
df_pos['gross_revenue'] = df_pos.unit_price * df_pos.quantity
df_neg['returned_revenue'] = df_neg.unit_price * abs(df_neg.quantity)
gross_revenue = df_pos.groupby('customer_id').sum()['gross_revenue'].reset_index()
returned_revenue = df_neg.groupby('customer_id').sum()['returned_revenue'].reset_index()

# last_purchase (days)
l_purchase = df_pos.groupby('customer_id').max()['invoice_date'].reset_index() 
l_purchase['last_purchase'] = (df_pos.invoice_date.max() - l_purchase.invoice_date).dt.days

# frequency
frequency = df_pos.groupby('customer_id').invoice_no.nunique().reset_index()
frequency.columns = ['customer_id', 'frequency']

# merging into client df and creating features
df_cli = pd.merge(df_cli, gross_revenue, on = 'customer_id', how = 'left')
df_cli = pd.merge(df_cli, returned_revenue, on = 'customer_id', how = 'left')
df_cli = pd.merge(df_cli, mix, on = 'customer_id', how = 'left')
df_cli['last_purchase'] = pd.merge(df_cli, l_purchase, on = 'customer_id', how = 'left')['last_purchase']
df_cli = pd.merge(df_cli, frequency, on = 'customer_id', how = 'left')
df_cli['average_ticket'] = df_cli.gross_revenue/df_cli.frequency

# checking and filling nas in new df with 0
df_cli.isna().sum()
df_cli = df_cli.fillna(0)

# saving df
df_cli.to_csv(os.path.join(FT_DIR, 'ft_df.csv'), index = False)