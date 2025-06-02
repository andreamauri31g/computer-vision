import tensorflow as tf

DATASET_PATH = "datasets/"
BATCH_SIZE = 16

generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.1, rescale=1.0/255)

train_generator = generator.flow_from_directory(
    DATASET_PATH,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

test_generator = generator.flow_from_directory(
    DATASET_PATH,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print(train_generator.class_indices)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, padding="same", activation="relu", input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=4, strides=4))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=4, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=4, strides=4))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_generator, epochs=5)

predict_train = model.evaluate(train_generator)
predict_test = model.evaluate(test_generator)

print(f"Train -> {predict_train}")
print(f"Test -> {predict_test}")

model.save("modelcatdog.h5")
