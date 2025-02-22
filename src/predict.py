import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("models/digit_classifier.h5")

# Load MNIST test dataset
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test = X_test / 255.0  # Normalize
X_test = X_test.reshape(-1, 28, 28, 1)  # Reshape for CNN

# Pick a random test image
index = np.random.randint(0, len(X_test))
image = X_test[index]
true_label = y_test[index]

# Make a prediction
predictions = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(predictions)

# Display the image and prediction
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True Label: {true_label} | Predicted: {predicted_label}")
plt.axis("off")
plt.show()

# Print prediction probabilities
print(f"Prediction Probabilities: {predictions}")
