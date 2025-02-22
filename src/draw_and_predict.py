import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("models/digit_classifier.h5")

# Create a blank white canvas
canvas = np.ones((400, 400), dtype="uint8") * 255
drawing = False  # True when mouse is pressed

# Mouse callback function to draw on canvas
def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 10, (0, 0, 0), -1)  # Draw black circles
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Set up OpenCV window
cv2.namedWindow("Draw a Digit (0-9)")
cv2.setMouseCallback("Draw a Digit (0-9)", draw)

while True:
    cv2.imshow("Draw a Digit (0-9)", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):  # Clear the canvas
        canvas[:] = 255
    elif key == ord("p"):  # Predict the digit
        # Preprocess the drawn image
        img = cv2.resize(canvas, (28, 28))  # Resize to match MNIST size
        img = cv2.bitwise_not(img)  # Invert colors (black digit on white)
        img = img / 255.0  # Normalize pixel values
        img = img.reshape(1, 28, 28, 1)  # Reshape for CNN

        # Make a prediction
        predictions = model.predict(img)
        predicted_label = np.argmax(predictions)

        print(f"Predicted Digit: {predicted_label}")

    elif key == ord("q"):  # Quit program
        break

cv2.destroyAllWindows()
