import pandas as pd
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot as plt
import cPickle as pickle
import os

FILENAME = 'indata - lex ky temperature.csv'
BACKUP_NAME = 'out - clean_forecast.csv'
INTERVAL_WIDTH = 0.99

# Import and clean data.
df = pd.read_csv(FILENAME, comment='#')
df = df[['valid','tmpf']]
df.rename(inplace=True, columns={"valid": "ds", "tmpf": "y"})
df.y.replace(to_replace={'M':0.0}, inplace=True)
# df['y'] = np.log(df['tmpf'])
# df.to_csv('clean.csv')

# Train the model.
if not os.path.isfile(BACKUP_NAME):
	print 'TRAINING MODEL!'
	# Build model.
	m = Prophet(yearly_seasonality=True, daily_seasonality=True, interval_width=INTERVAL_WIDTH)
	# m.add_country_holidays(country_name='US')
	m.fit(df)
	# Predict the future.
	print 'PREDICTING!'
	future = m.make_future_dataframe(periods=0)
	forecast = m.predict(future)
	# Merge in the historical data.
	forecast['y'] = [float(x) for x in df['y']]
	forecast['outlier'] = [False for x in xrange(len(forecast))]
	# Backup the model.
	forecast.to_csv(BACKUP_NAME)
else:
	print 'LOADING MODEL BACKUP!'
	forecast = pd.read_csv(BACKUP_NAME)

# Outliers.
print 'FINDING OUTLIERS!'
for index, row in forecast.iterrows():
	yVal = float(row['y'])
	yHigh = float(row['yhat_upper'])
	yLow = float(row['yhat_lower'])
	if (yVal > yHigh) or (yVal < yLow):
		forecast.at[index,'outlier'] = True
print 'Interval Width for Outliers:', INTERVAL_WIDTH
print 'Frame Rows:', len(forecast.index)
print 'Percent Outlying:', float(forecast.outlier.sum()) / len(forecast.index)

# Plot creation and showing.
fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
forecast_t = [np.datetime64(x) for x in forecast['ds']]
inliers = [(r[1] if not r[0] else None) for r in zip(forecast['outlier'], forecast['y'])]
outliers = [(r[1] if r[0] else None) for r in zip(forecast['outlier'], forecast['y'])]
ax.plot(forecast_t, forecast['yhat'], ls='-', c='#0072B2', label='prediction')
ax.plot(forecast_t, forecast['y'], 'k.', label='measurement')
ax.plot(forecast_t, outliers, 'r.', label='outlier')
ax.fill_between(forecast_t, forecast['yhat_lower'], forecast['yhat_upper'], color='#0072B2', alpha=0.2, label='detection interval')
ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
ax.set_xlabel('ds')
ax.set_ylabel('y')
ax.legend()
plt.title('Outliers and Model for "' + FILENAME + '"')
fig.tight_layout()
plt.show()