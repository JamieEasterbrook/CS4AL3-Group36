import numpy as np
import pandas as pd
import glob
import os

files = glob.glob(os.path.join("weather", '*.csv'))
df_list = []
for f in files:
    df = pd.read_csv(f)
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)
merged_df_sorted = merged_df.sort_values(by=['Year', 'Month', 'Day', 'Time (LST)'], ascending=True)
merged_df_sorted_filtered = merged_df_sorted.dropna(
    subset=[
        'Weather',
        'Temp (°C)',
        'Rel Hum (%)',
        'Stn Press (kPa)',
        'Visibility (km)',
        'Wind Spd (km/h)',
        'Dew Point Temp (°C)',
        'Wind Dir (10s deg)'
    ]
)
merged_df_sorted_filtered = merged_df_sorted_filtered.drop(
    labels=[
        'Longitude (x)',
        'Latitude (y)',
        'Station Name',
        'Climate ID',
        'Date/Time (LST)',
        'Flag',
        'Temp Flag',
        'Dew Point Temp Flag',
        'Rel Hum Flag',
        'Wind Dir Flag',
        'Wind Spd Flag',
        'Visibility Flag',
        'Hmdx Flag',
        'Wind Chill Flag',
        'Precip. Amount Flag',
        'Hmdx',
        'Stn Press Flag',
    ],
    axis=1
)
merged_df.to_csv('data/merged_raw.csv', index=False)

print(merged_df.sample(3))
merged_df_sorted_filtered.to_csv('data/merged_filtered.csv', index=False)

# TODO: Transform Year, Month, Date, Time, into features:
    # dtHours: Delta time in hours since the last measurement, 0 for the first one
    # sinTime: Local time transformed into a sinusoidal function (p=24)
    # sinDay: Month and Day transformed into days 0-365 then into a sinusoidal function (p=365)
    # year: Keep as it is
merged_final = merged_df_sorted_filtered