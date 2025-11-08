# Labels

## Terminology

- **Label**: Weather label, comes directly from the weather data and identifies unique ones.
- **Class**: A dimension/element of the output. Has a name and a range of real or integer values.
- **Section**: Defined by a combination of classes. Each combination of class values has a unique (or non-existant) label.

## Label Classes

All labels in a section should be exclusive from each other by definition (there is no combination of values that satisfies two labels in a section).

Cloudiness(Real, [0, 1]):

- Clear [0.00, 0.05)
- Mainly Clear [0.05, 0.10)
- Cloudy [0.10, 0.50)
- Mostly Cloudy [0.50, 1.00]

Precipitation(Real, [0, 1]), Intensity(Real, [0, 1]), Heat(Real, [0, 1])

- Drizzle [0.05, 0.10), [0.00, 0.50), [0.60, 1.00]
- Rain [0.10, 0.40), [0.00, 0.50), [0.60, 1.00]
- Moderate Rain [0.40, 0.60), [0.00, 0.50), [0.60, 1.00]
- Heavy Rain [0.60, 1.00], [0.00, 0.50), [0.60, 1.00]

- Rain Showers [0.10, 0.40), [0.50, 1.00], [0.60, 1.00]
- Moderate Rain Showers [0.40, 0.60), [0.50, 1.00], [0.60, 1.00]
- Heavy Rain Showers [0.60, 1.00], [0.50, 1.00], [0.60, 1.00]

- Freezing Drizzle [0.05, 0.10), [0.00, 0.50), [0.40, 0.60)
- Freezing Rain [0.10, 0.40), [0.00, 0.50), [0.40, 0.60)

- Snow [0.10, 0.40), [0.00, 0.50), [0.00, 0.40)
- Moderate Snow [0.40, 0.60), [0.00, 0.50), [0.00, 0.40)
- Heavy Snow [0.60, 1.00], [0.00, 0.50), [0.00, 0.40)

- Snow Showers [0.10, 1.00], [0.50, 1.00], [0.00, 0.50)

Heat(Real, [0, 1]), Visibility(Real, [0, 1])

- Fog [0.50, 1.00], [0.00, 0.50)
- Freezing Fog [0.00, 0.50), [0.00, 0.50)

Precipitation(Real, [0, 1]), Wind(Real, [0, 1]), Heat(Real, [0, 1])

- Blowing Snow [0.60, 1.00], [0.80, 1.00], [0.00, 0.40)
- Ice Pellets [0.60, 1.00], [0.80, 1.00], [0.45, 0.55)
- Snow Grains [0.60, 1.00], [0.60, 0.80), [0.40, 0.45)
- Snow Pellets [0.60, 1.00], [0.80, 1.00], [0.40, 0.45)
- Moderate Hail [0.60, 1.00], [0.80, 1.00], [0.40, 0.45)

Thunderstorms(Integer, [0, 1])

- Thunderstorms 1

Smoke(Integer, [0, 1])

- Smoke 1

Pollution(Real, [0, 1])

- Haze (0.50, 1.00]
