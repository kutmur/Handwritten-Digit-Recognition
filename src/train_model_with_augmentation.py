import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load MNIST dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape to (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create data augmentation pipeline - appropriate for handwritten digits
# We use modest augmentations that don't distort the digits beyond recognition
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),           # Slight rotation (±10%)
    tf.keras.layers.RandomZoom(0.1),               # Small zoom in/out
    tf.keras.layers.RandomTranslation(0.1, 0.1),   # Small horizontal/vertical shifts
    tf.keras.layers.RandomBrightness(0.1),         # Slight brightness adjustments
    # We avoid excessive distortions that could make digits unrecognizable
])

# Show examples of augmented images
def display_augmented_samples(X_samples, num_augmentations=5):
    """Display original and augmented samples for visualization"""
    fig = plt.figure(figsize=(12, 8))
    for i, img in enumerate(X_samples[:5]):
        # Original image
        ax = fig.add_subplot(5, 1+num_augmentations, i*(1+num_augmentations)+1)
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title('Original')
        ax.axis('off')
        
        # Augmented versions
        for j in range(num_augmentations):
            augmented_img = data_augmentation(np.expand_dims(img, axis=0))
            ax = fig.add_subplot(5, 1+num_augmentations, i*(1+num_augmentations)+j+2)
            ax.imshow(augmented_img[0].numpy().squeeze(), cmap='gray')
            ax.set_title(f'Aug #{j+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.close()

print("Creating augmentation examples...")
display_augmented_samples(X_train[:5])
print("Augmentation examples saved to 'augmentation_examples.png'")

# Create the CNN model with improved architecture
def create_model():
    model = tf.keras.models.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional layer
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create model
print("Creating model...")
model = create_model()
model.summary()

# Callbacks for training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
]

# Prepare datasets
print("Setting up data pipeline with augmentation...")

# Train with augmentation: Create a dataset with both original and augmented data
batch_size = 128
epochs = 20

# Create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Apply augmentation to the training data
def augment_data(images, labels):
    # Apply augmentation with 50% probability to avoid over-augmentation
    augment_probability = 0.5
    mask = tf.random.uniform(shape=[tf.shape(images)[0]], minval=0, maxval=1) < augment_probability
    
    augmented_images = tf.where(
        tf.reshape(mask, [-1, 1, 1, 1]),
        data_augmentation(images, training=True),
        images
    )
    
    return augmented_images, labels

# Apply the augmentation function and prefetch for performance
train_dataset = train_dataset.map(
    augment_data, 
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Test dataset (no augmentation needed for test data)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Train the model
print("Training model with data augmentation...")
start_time = datetime.now()
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)
end_time = datetime.now()

print(f"Training completed in {end_time - start_time}")

# Save the model
model.save("models/digit_classifier_augmented.h5")
print("Model saved to models/digit_classifier_augmented.h5")

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history_augmented.png')
plt.close()
print("Training history plot saved to 'training_history_augmented.png'")

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Create a comparison script to evaluate performance gain
with open("augmentation_performance_report.txt", "w") as f:
    f.write("# Handwritten Digit Recognition - Augmentation Performance Report\n\n")
    f.write(f"Test Accuracy with Augmentation: {test_acc*100:.2f}%\n")
    f.write(f"Test Loss with Augmentation: {test_loss:.4f}\n\n")
    f.write("## Data Augmentation Techniques Applied:\n")
    f.write("- Random Rotation (±10%)\n")
    f.write("- Random Zoom (±10%)\n")
    f.write("- Random Translation (±10%)\n")
    f.write("- Random Brightness Adjustments (±10%)\n\n")
    f.write("## Model Architecture Enhancements:\n")
    f.write("- Added BatchNormalization layers\n")
    f.write("- Improved dropout strategy\n")
    f.write("- Added learning rate scheduling\n")
    f.write("- Used early stopping to prevent overfitting\n\n")
    f.write("Check 'augmentation_examples.png' to see examples of the applied augmentations.\n")

print("Performance report saved to 'augmentation_performance_report.txt'")
print("Done!")