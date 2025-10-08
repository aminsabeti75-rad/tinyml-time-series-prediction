import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample time-series data
time = np.arange(0, 20).reshape(-1, 1)
values = 3 * time.squeeze() + np.random.randn(20) * 2  # y = 3x + noise

# Train model
model = LinearRegression()
model.fit(time, values)

# Predict next few steps
future_time = np.arange(20, 25).reshape(-1, 1)
predictions = model.predict(future_time)

# Plot
plt.scatter(time, values, label="Training data")
plt.plot(future_time, predictions, color="red", label="Predictions")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("TinyML-style Time-Series Prediction (Lightweight Model)")
plt.show()
