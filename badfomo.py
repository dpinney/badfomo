import pandas as pd
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot as plt
import cPickle as pickle
import os

# Import and clean data.
df = pd.read_csv('indata - lex ky.csv', comment='#')
df = df[['valid','tmpf']]
df.rename(inplace=True, columns={"valid": "ds", "tmpf": "y"})
df.y.replace(to_replace={'M':0.0}, inplace=True)
# df['y'] = np.log(df['tmpf'])
# df.to_csv('clean.csv')

# Train the model.
pickleName = 'trained lex ky temp model.pickle'
if not os.path.isfile(pickleName):
	print 'TRAINING MODEL!'
	# Build model.
	INTERVAL_WIDTH = 0.95
	m = Prophet(yearly_seasonality=True, daily_seasonality=True, interval_width=INTERVAL_WIDTH)
	m.add_country_holidays(country_name='US')
	m.fit(df)
	# Back up the model.
	with open(pickleName, 'wb') as pickFile:
		pickle.dump(m, pickFile)
else:
	print 'UNPICKLING MODEL!'
	with open(pickleName, 'rb') as pickFile:
		m = pickle.load(pickFile)

# Predict the future.
print 'PREDICTING!'
future = m.make_future_dataframe(periods=0)
forecast = m.predict(future)
# print forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# forecast.to_csv('clean_forecast.csv')

# Outliers.
overLiers = []
underLiers = []
for index, row in forecast.iterrows():
	yVal = float(df['y'][index])
	yHigh = float(row['yhat_upper'])
	yLow = float(row['yhat_lower'])
	if yVal > yHigh:
		overLiers.append(row)
	elif yVal < yLow:
		underLiers.append(row)
print 'Frame Rows:', len(forecast.index)
print 'Outliers:', len(overLiers) + len(underLiers)
print 'Interval Width for Outliers:', INTERVAL_WIDTH
print 'Percent Outlying:', float(len(forecast.index)) / float(len(overLiers) + len(underLiers))

# Show a forecast.
print 'PLOTTING!'
m.plot(forecast)
plt.show()

# Show components of forecast.
m.plot_components(forecast);
plt.show()