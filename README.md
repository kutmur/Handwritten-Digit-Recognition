#  Handwritten Digit Recognition (MNIST) 

## Overview  
This project is a **deep learning-based handwritten digit classifier** that recognizes digits **(0-9)** using a **Convolutional Neural Network (CNN)**. The model has been trained on the **MNIST dataset** and achieves an impressive **99.10% accuracy**.  

## 📌 Features  
- 🧠 **Trains a CNN model** to recognize handwritten digits  
- 🎨 **Interactive drawing tool** to test real-time digit prediction  
- 🏆 **99.10% test accuracy** on the MNIST dataset  
- 📊 **Graphical visualization** of model performance  
- 🖥️ **Predicts handwritten digits** from user input  

## 📂 Dataset  
This project uses the **MNIST dataset**, which consists of:  
👉 **60,000 training images**  
👉 **10,000 test images**  
👉 **Each image is a 28x28 grayscale digit (0-9)**  

The dataset is automatically downloaded using TensorFlow.  

## 🔧 Installation & Setup  
### 1. Clone the Repository  
```bash  
git clone https://github.com/EXPERT2007/Handwritten-Digit-Recognition.git  
cd Handwritten-Digit-Recognition  
```

### 2. Install Dependencies  
```bash  
pip install -r requirements.txt  
```

### 3. Train the CNN Model  
```bash  
python src/train_model.py  
```
✔️ Trains the CNN model on the MNIST dataset  
✔️ Saves the trained model to the `models/` directory  

### 4. Test on MNIST Dataset  
```bash  
python src/predict.py  
```
✔️ Loads a random image from MNIST and predicts the digit  
✔️ Displays the image and prints the predicted digit  

### 5. Interactive Drawing & Prediction  
```bash  
python src/draw_and_predict.py  
```
✔️ **Press 'p'** → Predict the drawn digit  
✔️ **Press 'c'** → Clear the canvas  
✔️ **Press 'q'** → Quit  

## 📊 Model Performance  
✔️ **Training Accuracy:** 99.49%  
✔️ **Test Accuracy:** 99.10%  
✔️ **CNN Model:** 2 Conv Layers + MaxPooling + Dense Layers  

## 📚 Project Structure  
```
Handwritten-Digit-Recognition/
│── data/                   # Stores dataset (if needed)
│── models/                 # Trained CNN model
│── notebooks/              # Jupyter Notebook for EDA
│── src/                    # Python scripts for training & prediction
│   │── train_model.py      # CNN training script
│   │── predict.py          # Tests model on MNIST images
│   │── draw_and_predict.py # Interactive digit drawing tool
│── README.md               # Project documentation
│── requirements.txt        # Dependencies
│── .gitignore              # Ignores unnecessary files
│── LICENSE                 # MIT License file
```

## 🛠️ Future Improvements  
- 📲 **Deploy as a web app** (Flask/Streamlit)  
- 🎨 **Enhance OpenCV UI** (Improve the digit drawing experience)  
- 📝 **Test with real-world handwriting samples**  

## 🤝 Contributing  
Feel free to fork this repo and submit pull requests. Contributions are welcome!  

## 📄 License  
This project is licensed under the MIT License - see the LICENSE file for details.  

---  


