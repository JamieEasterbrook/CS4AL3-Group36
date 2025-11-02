import pandas as pd
import numpy as np

# Find longest sequence of 3-hour data
# TODO: Determine if interpolation should be used and figure out preprocessing details

def days_in_month(year:int, month:int):
    days_reg = [
    -1,
    31,
    28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
    ]
    if year % 4 != 0 & month != 2:
        return days_reg[month]
    else:
        return 29

def get_next_point(year:int, month:int, day:int, hour:int) -> tuple[int,int,int,int]:
    if hour <= 19:
        return (year,month,day,hour+3)
    else:
        if day < days_in_month(year,month):
            return (year, month, day+1, (hour+3) % 24)
        else:
            if month < 12:
                return (year, month+1, 1, (hour+3) % 24)
            else:
                return (year+1, 1, 1, (hour+3) % 24)

df: pd.DataFrame = pd.read_csv("data/merged_filtered.csv")
nd: np.ndarray = df.to_numpy()
columns = {
    "Year":0,
    "Month": 1,
    "Day": 2,
    "Time (LST)": 3,
    "Temp (°C)": 4,
    "Dew Point Temp (°C)":5,
    "Rel Hum (%)":6,
    "Wind Dir (10s deg)":7,
    "Wind Spd (km/h)":8,
    "Visibility (km)":9,
    "Stn Press (kPa)":10,
    "Wind Chill":11,
    "Weather":12,
    "Precip. Amount (mm)":13
}

intervals:list[tuple[int,int]] = []
i = 0
j = 0
previous = nd[0]
for current in nd[1:]:
    next_expected = get_next_point(
        previous[columns["Year"]], 
        previous[columns["Month"]], 
        previous[columns["Day"]], 
        int(str(previous[columns["Time (LST)"]])[:2])
    )
    
    if next_expected != (
        current[columns["Year"]], 
        current[columns["Month"]], 
        current[columns["Day"]], 
        int(str(current[columns["Time (LST)"]])[:2])
    ):
        intervals.append((i,j))
        i = j

    j+=1
    previous = current
