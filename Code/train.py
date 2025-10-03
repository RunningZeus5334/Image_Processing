from data_loader import load_data
from cnn_model import create_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------
# Data Augmentation Setup
# ------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator, test_generator = load_data()

# ------------------------
# Create Model
# ------------------------
model = create_model()

# ------------------------
# Callbacks for training
# ------------------------
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "best_emotion_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ------------------------
# Train Model
# ------------------------
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    callbacks=[reduce_lr, checkpoint]
)

# ------------------------
# Save Final Model
# ------------------------
model.save("emotion_model_final.h5")
print("✅ Model saved as emotion_model_final.h5")
