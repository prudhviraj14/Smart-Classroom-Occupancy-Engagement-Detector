import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Image settings
img_size = (48, 48)
batch_size = 64
num_classes = 5  # Angry, Happy, Sad, Fear, Surprise, Neutral, Sleep

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

# ✅ CNN model for FER
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100
)

# Evaluate
test_loss, test_acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# Save model
model.save("emotion_model_7class.h5")
print("💾 Model saved as emotion_model_7class.h5")
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # Paths
# train_dir = "dataset/train"
# val_dir = "dataset/val"
# test_dir = "dataset/test"

# # Settings
# img_size = (48, 48)
# batch_size = 64
# num_classes = 7  # angry, fear, happy, neutral, sad, sleep, surprise

# # ✅ Data Augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=25,
#     width_shift_range=0.25,
#     height_shift_range=0.25,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# val_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_gen = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=img_size,
#     color_mode="grayscale",
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=True
# )

# val_gen = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=img_size,
#     color_mode="grayscale",
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=False
# )

# test_gen = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=img_size,
#     color_mode="grayscale",
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=False
# )

# # ✅ CNN Model
# model = Sequential([
#     Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),
#     Dropout(0.25),

#     Conv2D(128, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),
#     Dropout(0.25),

#     Conv2D(256, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),
#     Dropout(0.25),

#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# # ✅ Compile
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])

# # ✅ Callbacks
# early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
# checkpoint = ModelCheckpoint("best_emotion_model.h5", monitor="val_accuracy", save_best_only=True, mode="max")

# # ✅ Train
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=60,
#     callbacks=[early_stop, checkpoint],
#     verbose=1
# )

# # ✅ Evaluate on test set
# test_loss, test_acc = model.evaluate(test_gen)
# print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# # ✅ Save final model
# model.save("emotion_model_7class.h5")
# print("💾 Final model saved as emotion_model_7class.h5")
