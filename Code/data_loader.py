from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "train",
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        "test",
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical"
    )

    return train_generator, test_generator
