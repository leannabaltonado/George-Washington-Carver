#%pip install -r hackingfood_requirements.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cfe.regression as rgsn

Uganda_Data = '1yFWlP5N7Aowaj6t2roRSFFUC50aFD-RLBGfzGtqLl0w'
NG_URL= "https://docs.google.com/spreadsheets/d/1rbWPdn_Rt3MRoFb_wkomHBcNVmM-3F3lNwCpxP5W_3g/edit?gid=1354121303#gid=1354121303" 
from ligonlibrary.sheets import read_sheets


# 1. LOAD DATA
x_raw = read_sheets(NG_URL, sheet='Food Expenditures (2016Q1)')
d_raw = read_sheets(NG_URL, sheet="Household Characteristics")

# 2. STANDARDIZE THE KEYS (The Fix)
def clean_keys(df):
    df = df.copy()
    # Force 'i' to be numeric (removes string/int mismatch)
    df['i'] = pd.to_numeric(df['i'], errors='coerce')
    # Force 't' to be a clean string
    df['t'] = df['t'].astype(str).str.strip()
    # Force 'm' to be lowercase/stripped string
    df['m'] = df['m'].astype(str).str.strip().str.lower()
    return df.dropna(subset=['i'])

x_clean = clean_keys(x_raw)
d_clean = clean_keys(d_raw)

# 3. SET INDICES AND COLLAPSE DUPLICATES
# Expenditures (y)
x = x_clean.groupby(['i', 't', 'm', 'j'])['Expenditure'].sum()
y = np.log(x.replace(0, np.nan))

# Characteristics (d)
# Use .first() to ensure one row per household/time/market
d = d_clean.groupby(['i', 't', 'm']).first() 
d = d.loc[:, ~d.columns.duplicated()].fillna(0)

# 4. RUN REGRESSION (Should work now)
r = rgsn.Regression(y=y, d=d)


# Now prices
p = read_sheets(NG_URL,sheet='Food Prices (2016Q1)').set_index(['t','m','j','u'])

# Compute medians of prices for particular time, place and unit
p = p.groupby(['t','m','j','u']).median()

# Just keep metric units
p = p.xs('kg',level="u").squeeze().unstack('j')

# Get intersection of goods we have prices *and* expenditures for:
jidx = p.columns.intersection(x.index.levels[-1])

# Drop prices for goods we don't have expenditures for
p = p[jidx].T


# Assumes you've already set this up e.g., in Project 3
# r = rgsn.read_pickle('../Project3/uganda_estimates.rgsn')

# Load the CSV
fct = pd.read_csv('FoodNutrientsNG.csv')

# 1. Rename the first column to 'j' so we can use it as our food index
# 2. Set it as the index
# 3. Name the columns 'n' (for Nutrients)
fct = fct.rename(columns={'Unnamed: 0': 'j'}).set_index('j')
fct.columns.name = 'n'

# Convert all nutrient values to numbers (forcing errors to NaN)
fct = fct.apply(lambda x: pd.to_numeric(x, errors='coerce'))


# 1. Load the RDI CSV from the web
rdi_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vR84qu7yUvdzhiHnPG_7KpVSx9FgiyLrZqJ8Ft3oK-B6xT6i4xN8CNtZ2c9PRGXqxXQCNorzWdF6NnB/pub?gid=0&single=true&output=csv'
rdi = pd.read_csv(rdi_url)

# 2. Rename the first column (whatever it is) to 'n' and set as index
rdi = rdi.rename(columns={rdi.columns[0]: 'n'}).set_index('n')

# 3. Set the columns name to 'k' (representing household characteristic categories)
rdi.columns.name = 'k'

# 4. Force values to be numeric
rdi = rdi.apply(lambda x: pd.to_numeric(x, errors='coerce'))
# Reference prices chosen from a particular time; average across place.
# These are prices per kilogram:
# NB: fillna(1) replaces missing prices with 1 (currency unit per kg).
# This is a rough default so that the code runs; in a serious analysis
# you would want to investigate *which* goods lack price data and
# either find prices from another source or drop those goods.
pbar = p.loc[r.get_beta().index].mean(axis=1).fillna(1)

xhat = r.predicted_expenditures()

# Total food expenditures per household
xbar = xhat.groupby(['i','t','m']).sum()

# Reference budget
xref = xbar.quantile(0.5)  # Household at 0.5 quantile is median

qhat = (xhat.unstack('j')/pbar).dropna(how='all')

# Drop missing columns
qhat = qhat.loc[:,qhat.count()>0]

qhat

def ceteris_paribus_price(j,p0,p=pbar):
    """
    Return price vector with the price of good j set to p0,
    holding all other prices fixed at p.
    """
    p = p.copy()
    p.loc[j] = p0
    return p

fct

# Create a new FCT and vector of consumption that only share rows in common:
fct0,c0 = fct.align(qhat.T,axis=0,join='inner')
print(fct0.index)

# The @ operator means matrix multiply
N = fct0.T@c0

N  #NB: Uganda quantities are for previous 7 days

def nutrient_demand(x,p,r=r,fct=fct):
    c = r.demands(x,p)
    fct0,c0 = fct.align(c,axis=0,join='inner')
    N = fct0.T@c0

    # Drop duplicate nutrient rows (keeps first).  If your FCT has
    # duplicates this silently discards data; worth checking with
    # fct.index[fct.index.duplicated()] to see what's being dropped.
    N = N.loc[~N.index.duplicated()]

    return N

# In first round, averaged over households and villages
dbar = r.d[rdi.columns].mean()

# This matrix product gives minimum nutrient requirements for
# the average household
hh_rdi = rdi@dbar

hh_rdi

def nutrient_adequacy_ratio(x,p,d,rdi=rdi,days=7):
    hh_rdi = rdi.replace('',0)@d*days

    return nutrient_demand(x,p)/hh_rdi