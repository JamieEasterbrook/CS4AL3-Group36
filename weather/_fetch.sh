#!/bin/bash

# Running this script should not be necessary as the data is already in data/merged_filtered.csv
for year in `seq 1950 2025`;do for month in `seq 1 12`;do wget --content-disposition "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=51459&Year=${year}&Month=${month}&Day=14&timeframe=1&submit=Download+Data" ;done;done;