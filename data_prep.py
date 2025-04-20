import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('hyundi.csv')
df.head()

df.info()

df.isnull().sum()

# treat outlier in the price variable

# Calculate the interquartile range (IQR)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Define the outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Apply capping
df['price'] = df['price'].clip(lower=lower_bound, upper=upper_bound)

df.to_csv('hyundi_processed.csv', index=False)


