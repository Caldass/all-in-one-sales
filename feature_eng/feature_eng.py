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

# turning columns from camel case to snake case
snakecase = lambda x:inflection.underscore(x)
cols_new = list(map(snakecase, df.columns))
df.columns = cols_new



## Data Description

# data types and df shape
df.dtypes
df.shape

# check na
df.isna().sum()

df_missing = df[df.customer_id.isna()]
df_not_missing = df[df.customer_id.notna()]

# create reference
df_backup = pd.DataFrame( df_missing['invoice_no'].drop_duplicates() )
df_backup['customer_id'] = np.arange( 19000, 19000+len( df_backup ), 1)

# merge original with reference dataframe
df = pd.merge( df, df_backup, on='invoice_no', how='left' )

# coalesce 
df['customer_id'] = df['customer_id_x'].combine_first( df['customer_id_y'] )

# drop extra columns
df.drop( columns=['customer_id_x', 'customer_id_y'] , inplace = True)
df.head()

# data types
df.dtypes
df.shape

# changing dtypes
df['invoice_date'] = pd.to_datetime(df.invoice_date)
df['customer_id'] = df.customer_id.astype(int)



## Descriptive Statistics
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

# filtering out stock code containing strings
df = df[~cat_attrs.stock_code.apply(lambda x: bool(re.search('^[a-zA-Z]+$', x)) )]

# removing price equal to 0
df = df[df['unit_price'] >= 0.04]

# creating revenue and return dataframe
df_purchase = df[df.quantity >= 0]
df_returns = df[df.quantity < 0]

# removing countries not identified and description column
df = df[~df['country'].isin( ['European Community', 'Unspecified' ] ) ]
df.drop(columns = 'description', inplace = True)

# removing bad users 
df = df[~df.customer_id.isin( [16446] )]



## Feature Engineering

# distinct client df
df_cli = df[['customer_id']].drop_duplicates(ignore_index = True)

 # avg recency days
df_aux = df[['customer_id', 'invoice_date']].drop_duplicates().sort_values( ['customer_id', 'invoice_date'], ascending=['False', 'False'] )
df_aux['next_customer_id'] = df_aux['customer_id'].shift() # next customer
df_aux['previous_date'] = df_aux['invoice_date'].shift() # next invoice date
df_aux['avg_recency_days'] = df_aux.apply( lambda x: ( x['invoice_date'] - x['previous_date'] ).days if x['customer_id'] == x['next_customer_id'] else np.nan, axis=1 )
df_aux = df_aux.drop( ['invoice_date', 'next_customer_id', 'previous_date'], axis=1 ).dropna()

df_avg_recency_days = df_aux.groupby( 'customer_id' ).mean().reset_index()

df_cli = pd.merge(df_cli, df_avg_recency_days, on = 'customer_id', how = 'left')


# basket_size
df_aux = ( df_purchase[['customer_id', 'invoice_no', 'quantity']].groupby( 'customer_id' )
                                                                            .agg( n_purchase=( 'invoice_no', 'nunique'),
                                                                                  n_products=( 'quantity', 'sum' ) )
                                                                            .reset_index() )
df_aux['avg_basket_size'] = df_aux['n_products'] / df_aux['n_purchase']
basket_size = df_aux[['customer_id','avg_basket_size']]

df_cli = pd.merge(df_cli, basket_size, on = 'customer_id', how = 'left')


# unique basket size
df_aux = ( df_purchase[['customer_id', 'invoice_no', 'stock_code']].groupby( 'customer_id' )
                                                                            .agg( n_purchase=( 'invoice_no', 'nunique'),
                                                                                   n_products=( 'stock_code', 'nunique' ) )
                                                                            .reset_index() )

df_aux['avg_unique_basket_size'] = df_aux['n_products'] / df_aux['n_purchase']
unique_basket_size = df_aux[['customer_id', 'avg_unique_basket_size']]

df_cli = pd.merge(df_cli, unique_basket_size, on = 'customer_id', how = 'left')


# revenue and returned amount
df_purchase['gross_revenue'] = df_purchase.unit_price * df_purchase.quantity
df_returns['returned_revenue'] = df_returns.unit_price * abs(df_returns.quantity)
gross_revenue = df_purchase.groupby('customer_id').sum()['gross_revenue'].reset_index()
#returned_revenue = df_returns.groupby('customer_id').sum()['returned_revenue'].reset_index()


df_cli = pd.merge(df_cli, gross_revenue, on = 'customer_id', how = 'left')
#df_cli = pd.merge(df_cli, returned_revenue, on = 'customer_id', how = 'left')


# Number of Returns
df_returns = df_returns[['customer_id', 'quantity']].groupby( 'customer_id' ).sum().reset_index().rename( columns={'quantity':'qt_returns'} )
df_returns['qt_returns'] = df_returns['qt_returns'] * -1

df_cli = pd.merge( df_cli, df_returns, how='left', on='customer_id' )
df_cli.loc[df_cli['qt_returns'].isna(), 'qt_returns'] = 0


# last_purchase (days)
l_purchase = df_purchase.groupby('customer_id').max()['invoice_date'].reset_index() 
l_purchase['last_purchase'] = (df_purchase.invoice_date.max() - l_purchase.invoice_date).dt.days

df_cli = pd.merge(df_cli, l_purchase[['customer_id', 'last_purchase']], on = 'customer_id', how = 'left')


# number of orders
orders = df_purchase.groupby('customer_id').invoice_no.nunique().reset_index()
orders.columns = ['customer_id', 'orders']

df_cli = pd.merge(df_cli, orders, on = 'customer_id', how = 'left')


# Qt of products purchases
qt_products = (df_purchase[['customer_id', 'stock_code']].groupby( 'customer_id' ).count()
                                                           .reset_index()
                                                           .rename( columns={'stock_code': 'qt_products'} ) )
df_cli = pd.merge( df_cli, qt_products, on='customer_id', how='left' )


# total items purchases
total_items = (df_purchase[['customer_id', 'quantity']].groupby( 'customer_id' ).sum()
                                                           .reset_index()
                                                           .rename( columns={'quantity': 'qt_items'} ) )
df_cli = pd.merge( df_cli, total_items, on='customer_id', how='left' )


# frequency of purchases
df_aux = ( df_purchase[['customer_id', 'invoice_no', 'invoice_date']].drop_duplicates()
                                                             .groupby( 'customer_id')
                                                             .agg( max_ = ( 'invoice_date', 'max' ), 
                                                                   min_ = ( 'invoice_date', 'min' ),
                                                                   days_= ( 'invoice_date', lambda x: ( ( x.max() - x.min() ).days ) + 1 ),
                                                                   buy_ = ( 'invoice_no', 'count' ) ) ).reset_index()

df_aux['frequency'] = df_aux[['buy_', 'days_']].apply( lambda x: x['buy_'] / x['days_'] if  x['days_'] != 0 else 0, axis=1 )
freq = df_aux[['customer_id', 'frequency']]

df_cli = pd.merge( df_cli, freq, on='customer_id', how='left' )


# creating final column in client df
df_cli['average_ticket'] = df_cli.gross_revenue/df_cli.orders


# checking and filling nas in new df with 0
df_cli.isna().sum()
#df_cli = df_cli.fillna(0)



## Exporting to csv

# saving df
df_cli.to_csv(os.path.join(FT_DIR, 'ft_df.csv'), index = False)
df.to_csv(os.path.join(FT_DIR, 'full_df.csv'), index = False)