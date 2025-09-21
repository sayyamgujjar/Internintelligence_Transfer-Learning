import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory


# Step 1: Load a small dataset (example: Flowers dataset from TensorFlow)
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

# Create training and validation datasets
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(160, 160),
    batch_size=32)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(160, 160),
    batch_size=32)

# Step 2: Load Pre-trained MobileNetV2
base_model = keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                            include_top=False,
                                            weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Step 3: Add custom layers on top
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(5, activation='softmax')   # 5 flower categories
])

# Step 4: Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=3)

# Step 6: Fine-tuning (optional: unfreeze some layers for better accuracy)
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=2)

# Step 7: Evaluate
loss, acc = model.evaluate(val_ds)
print("Validation Accuracy:", acc)
