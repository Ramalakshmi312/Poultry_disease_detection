from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3), name="input_layer_1"))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# FIX: Use 128 instead of 64 to match saved weights
model.add(Dense(128, activation='relu'))

# Output layer — keep number of classes same (adjust if needed)
model.add(Dense(3, activation='softmax'))

# Load weights
model.load_weights("poultry_disease_model.h5")

# Save the model in .keras format
model.save("poultry_disease_model.keras")
