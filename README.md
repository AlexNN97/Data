# Data
Ongoing learning of Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

# Original setup
carsales['sin'] = carsales['period'].apply(lambda x: math.sin(x*2*math.pi/12))
carsales['cos'] = carsales['period'].apply(lambda x: math.cos(x*2*math.pi/12))

x_trig = carsales.loc[:,['period','sin','cos']].values.reshape(-1,3)
y = carsales['sales'].values.reshape(-1,1)
triginometry_regression = LinearRegression()
triginometry_regression.fit(x_trig,y)
trig_line = triginometry_regression.predict(x_trig)[:,0]
saleslist = carsales['sales'].tolist()

# Extend the x values (for example, up to period 150)
extended_periods = np.arange(carsales['period'].max() + 1, 151)
sin_extended = [math.sin(x*2*math.pi/12) for x in extended_periods]
cos_extended = [math.cos(x*2*math.pi/12) for x in extended_periods]

# Combine the original with the extended values
all_periods = np.concatenate([carsales['period'].values, extended_periods])
all_sin = np.concatenate([carsales['sin'].values, sin_extended])
all_cos = np.concatenate([carsales['cos'].values, cos_extended])

x_trig_extended = np.column_stack([all_periods, all_sin, all_cos])

# Get predictions for the extended set
trig_line_extended = triginometry_regression.predict(x_trig_extended)

# Plot
plt.plot(carsales['period'], carsales['sales'], label="Actual Sales")
plt.plot(all_periods, trig_line_extended, 'r--', label="Trig Predictions")
plt.legend()
plt.show()
