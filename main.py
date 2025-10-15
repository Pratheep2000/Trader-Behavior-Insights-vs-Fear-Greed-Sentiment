# Runtime installs (only if needed)
%pip install -q gdown
%pip install -q matplotlib==3.7.1 pandas==2.2.2 seaborn==0.12.2

import os
import io
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create folders
os.makedirs("C:\\Users\\dines\\Downloads", exist_ok=True)
open("C:\\Users\\dines\\Downloads\\historical_data.csv", 'a').close()
open("C:\\Users\\dines\\Downloads\\fear_greed_index.csv", 'a').close()

sns.set(style='darkgrid')
%matplotlib inline

# Replace these with the actual Google Drive shareable links (from the assignment PDF):
trades_gdrive = 'https://drive.google.com/uc?id=1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs&export=download'
fear_greed_gdrive = 'https://drive.google.com/uc?id=1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf&export=download'

import gdown

# Download files to csv_files
trades_path = "C:\\Users\\dines\\Downloads\\historical_data.csv"
fear_path = "C:\\Users\\dines\\Downloads\\fear_greed_index.csv"

print('Downloading trades...')
gdown.download(trades_gdrive, trades_path, quiet=False)
print('Downloading fear/greed...')
gdown.download(fear_greed_gdrive, fear_path, quiet=False)

# Quick load
trades = pd.read_csv(trades_path)
fear = pd.read_csv(fear_path)

print('trades', trades.shape)
print('fear', fear.shape)



# Preview and initial cleaning (adapt as needed)
trades.head()

# Convert time columns to datetimes (try multiple formats)
if 'time' in trades.columns:
    trades['time'] = pd.to_datetime(trades['time'], errors='coerce')

# Clean numeric columns
for col in ['execution price', 'closedPnL', 'size', 'leverage']:
    if col in trades.columns:
        trades[col] = (trades[col].astype(str).str.replace(',', '').str.replace('$', '').replace({'': np.nan}) )
        trades[col] = pd.to_numeric(trades[col], errors='coerce')

# Parse fear greed date
if 'Date' in fear.columns:
    fear['Date'] = pd.to_datetime(fear['Date'], errors='coerce')

# Create a trade_date (date part only) for merging
if 'time' in trades.columns:
    trades['trade_date'] = trades['time'].dt.date
else:
    for c in trades.columns:
        if 'date' in c.lower():
            trades['trade_date'] = pd.to_datetime(trades[c], errors='coerce').dt.date
            break

# Convert fear Date to date
if 'Date' in fear.columns:
    fear['date_only'] = fear['Date'].dt.date

# Save cleaned copies
trades.to_csv("C:\\Users\\dines\\Downloads\\historical_data.csv", index=False)
fear.to_csv("C:\\Users\\dines\\Downloads\\fear_greed_index.csv", index=False)

trades.shape, fear.shape


# Feature engineering and aggregation
if 'closedPnL' in trades.columns:
    trades['is_profit'] = trades['closedPnL'] > 0
    trades['abs_pnl'] = trades['closedPnL'].abs()

if 'side' in trades.columns:
    trades['side'] = trades['side'].astype(str).str.lower()

# Daily trader aggregates
if 'account' in trades.columns and 'trade_date' in trades.columns:
    daily = trades.groupby(['account','trade_date']).agg(
        total_pnl = ('closedPnL','sum'),
        avg_pnl = ('closedPnL','mean'),
        win_rate = ('is_profit','mean'),
        total_volume = ('size','sum') if 'size' in trades.columns else ('execution price','count'),
        avg_leverage = ('leverage','mean') if 'leverage' in trades.columns else ('execution price','count')
    ).reset_index()
else:
    daily = pd.DataFrame()

# Aggregate to platform-level per date
if not daily.empty:
    platform_daily = daily.groupby('trade_date').agg(
        platform_total_pnl = ('total_pnl','sum'),
        platform_avg_win_rate = ('win_rate','mean'),
        platform_avg_leverage = ('avg_leverage','mean')
    ).reset_index()
else:
    platform_daily = pd.DataFrame()

# Merge with sentiment
if 'date_only' in fear.columns and not platform_daily.empty:
    platform_daily['date_only'] = pd.to_datetime(platform_daily['trade_date']).dt.date
    merged = platform_daily.merge(fear[['date_only','Classification']], on='date_only', how='left')
else:
    merged = platform_daily.copy()

# Ensure the output directory exists before saving
output_dir = "C:\\Users\\dines\\Downloads\\csv_files"
os.makedirs(output_dir, exist_ok=True)
merged.to_csv(os.path.join(output_dir, "merged_by_date.csv"), index=False)
merged.head()


# Exploratory plots
import matplotlib.dates as mdates

# Boxplot: platform_total_pnl by sentiment
plt.figure(figsize=(10,5))
if 'platform_total_pnl' in merged.columns and 'Classification' in merged.columns:
    sns.boxplot(x='Classification', y='platform_total_pnl', data=merged)
    plt.title('Platform total PnL distribution by Sentiment')
    plt.savefig('/content/outputs/eda_profit_vs_sentiment.png', bbox_inches='tight')
    plt.show()

# Avg leverage vs sentiment
plt.figure(figsize=(8,5))
if 'platform_avg_leverage' in merged.columns and 'Classification' in merged.columns:
    sns.boxplot(x='Classification', y='platform_avg_leverage', data=merged)
    plt.title('Platform average leverage by Sentiment')
    plt.savefig('/content/outputs/leverage_vs_sentiment.png', bbox_inches='tight')
    plt.show()

# Time series: 7-day rolling total PnL
if 'trade_date' in platform_daily.columns and 'platform_total_pnl' in platform_daily.columns:
    plt.figure(figsize=(12,5))
    plt.plot(pd.to_datetime(platform_daily['trade_date']), platform_daily['platform_total_pnl'].rolling(7).mean())
    plt.title('7-day rolling total PnL (platform)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.savefig('/content/outputs/rolling_pnl.png', bbox_inches='tight')
    plt.show()



# Simple statistical tests
from scipy import stats
if 'platform_total_pnl' in merged.columns and 'Classification' in merged.columns:
    fear_pnl = merged.loc[merged['Classification'].str.lower()=='fear','platform_total_pnl'].dropna()
    greed_pnl = merged.loc[merged['Classification'].str.lower()=='greed','platform_total_pnl'].dropna()
    print('Fear mean:', fear_pnl.mean(), 'Greed mean:', greed_pnl.mean())
    # t-test
    if len(fear_pnl)>1 and len(greed_pnl)>1:
        tstat, pval = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False, nan_policy='omit')
        print('t-stat:', tstat, 'p-value:', pval)

# Correlation table
corr_cols = [c for c in merged.columns if merged[c].dtype in [np.float64, np.int64]]
if corr_cols:
    display(merged[corr_cols].corr())



# Save processed files
os.makedirs(output_dir, exist_ok=True)
merged.to_csv(os.path.join(output_dir, 'merged_summary_by_date.csv'), index=False)
if 'daily' in globals() and not daily.empty:
    daily.to_csv(os.path.join(output_dir, 'daily_trader_aggregates.csv'), index=False)

print(f'Saved processed CSVs to {output_dir} and visual outputs to /content/outputs')
