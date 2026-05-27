import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# 1. Load/Download Dataset
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
print("Loading dataset...")
df = pd.read_csv(DATA_URL)

# 2. Preprocessing (matching Colab work)
X = df[['Pclass', 'Age', 'Fare']].copy()
y = df['Survived']

# Fill missing values
X['Age'] = X['Age'].fillna(X['Age'].mean())

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 3. Build ANN Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Train Model
print("Training model...")
try:
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        verbose=1
    )
    print("Training finished successfully.")
except Exception as e:
    print(f"Error during training: {e}")

# 5. Save Assets
print("Saving model and scaler...")
try:
    model.save("model/titanic_ann_model.keras")
    print("Model saved.")
    joblib.dump(scaler, "model/scaler.pkl")
    print("Scaler saved.")
    df.to_csv("data/Titanic-Dataset.csv", index=False)
    print("Dataset saved.")
except Exception as e:
    print(f"Error during saving: {e}")

print("Setup complete!")
