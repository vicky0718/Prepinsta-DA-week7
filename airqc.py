import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('weather_cleaned_data1.csv')

# print(data.head())
x = data[['rspm']]
y = data[['spm']]

x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state= 40)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 7: Use the model to predict PM2.5 values
# Replace 'your_rspm_values' with the actual RSPM values you want to predict PM2.5 for
your_rspm_values = pd.DataFrame({'rspm': [10, 20, 30]})  # Example RSPM values
predicted_pm2_5 = model.predict(your_rspm_values)
print(predicted_pm2_5)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

