import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Step 1: Generate Synthetic Dataset
"""
data = {
    'Muscle Contraction': np.random.uniform(0, 1, 100),  # Generate random muscle contraction data
    'Accelerometer Angle': np.random.uniform(0, 90, 100),  # Generate random accelerometer angle data
    'Load Cell Measurement': np.random.uniform(10, 30, 100),  # Generate random load cell measurement data
    'Motor Engage': np.random.randint(0, 2, 100)  # Generate random motor engage (target) data
}
"""
data=pd.read_csv('data.csv')
df = pd.DataFrame(data)

print(df)

# Step 2: Prepare Features (X) and Target (y)
X = df[['Muscle Contraction', 'Accelerometer Angle', 'Load Cell Measurement']]
y = df['Motor Engage']

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

sensor_data = pd.read_csv('sensordata.csv').drop(columns=['Motor Engage'])


# Step 6: Deployment - Use the trained model for real-time predictions
# Assuming you have real-time sensor data stored in variable 'sensor_data'
motor_engage = clf.predict(sensor_data)
print(motor_engage)
# Control the motor based on 'motor_engage' prediction