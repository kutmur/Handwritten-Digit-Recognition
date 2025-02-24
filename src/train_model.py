import cv2
import numpy as np
import tensorflow as tf
import time  # Used to detect when drawing stops

# Load the trained model
model = tf.keras.models.load_model("models/digit_classifier.h5")

# Increase display size
canvas_height, canvas_width = 500, 500  # Bigger drawing area
prediction_space_height = 150  # More space for predictions
full_canvas = np.ones((canvas_height + prediction_space_height, canvas_width), dtype="uint8") * 255

canvas = np.ones((canvas_height, canvas_width), dtype="uint8") * 255
drawing = False
predicted_label = None  # Store predicted digit
last_draw_time = None  # Track last drawing time
auto_predict_delay = 0.3  # Faster real-time prediction (0.3s)

last_canvas_state = None  # Track changes to avoid unnecessary predictions

# Mouse callback function
def draw(event, x, y, flags, param):
    global drawing, last_draw_time
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and y < canvas_height:  # Prevent drawing in the prediction area
            cv2.circle(canvas, (x, y), 12, (0, 0, 0), -1)
            last_draw_time = time.time()  # Update last draw time
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Function to predict the digit
def predict_digit():
    global predicted_label, last_canvas_state

    # Only predict if the canvas has changed
    if last_canvas_state is not None and np.array_equal(canvas, last_canvas_state):
        return  # Skip prediction if canvas is unchanged

    last_canvas_state = canvas.copy()  # Save current state

    img = cv2.resize(canvas, (28, 28))  # Resize to MNIST format
    img = cv2.bitwise_not(img)  # Invert colors (black digit on white)
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for CNN
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)  # Get highest confidence digit

# Set up OpenCV window
cv2.namedWindow("Handwritten Digit Recognition")
cv2.setMouseCallback("Handwritten Digit Recognition", draw)

while True:
    # Copy full canvas and add the drawing part
    full_canvas[:canvas_height, :] = canvas

    # Clear the prediction space
    full_canvas[canvas_height:, :] = 255

    # Draw a separation line
    cv2.line(full_canvas, (0, canvas_height), (canvas_width, canvas_height), (0, 0, 0), 3)

    # **REAL-TIME PREDICTION**: Check if the user is drawing, and predict instantly
    if last_draw_time and (time.time() - last_draw_time > auto_predict_delay):
        predict_digit()
        last_draw_time = time.time()  # Reset the timer to avoid excessive predictions

    # Display prediction if available
    if predicted_label is not None:
        # Center the text horizontally
        text_label = "Prediction:"
        text_size = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        label_x = (canvas_width - text_size[0]) // 2

        text_digit = str(predicted_label)
        digit_size = cv2.getTextSize(text_digit, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
        digit_x = (canvas_width - digit_size[0]) // 2  # Center the digit

        # Display the text in the center
        cv2.putText(full_canvas, text_label, (label_x, canvas_height + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)  # Blue text

        cv2.putText(full_canvas, text_digit, (digit_x, canvas_height + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 0), 5)  # Green text

    # Show the complete window
    cv2.imshow("Handwritten Digit Recognition", full_canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):  # Clear canvas
        canvas[:] = 255
        predicted_label = None  # Reset prediction
        last_draw_time = None  # Reset auto-predict timer
    elif key == ord("q"):  # Quit
        break

cv2.destroyAllWindows()
