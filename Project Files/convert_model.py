from tensorflow.keras.models import load_model

# Step 1: Load your existing .h5 model
model = load_model("poultry_disease_model.h5")

# Step 2: Save it again in the newer .keras format
model.save("poultry_disease_model.keras")

# Optional: Save a fresh .h5 version (cleaned up)
model.save("poultry_disease_model_updated.h5")
