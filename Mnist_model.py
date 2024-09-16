import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np

#  Loading the MNIST dataset and will be using to train
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data preprocessing
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# Normalize the data values to range between 0 and 1
x_train /= 255
x_test /= 255

# one-hot encoding for value conversion
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# building the convolutional neural net
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

# model compilation stage
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200)

model.save('mnist_digit_recognition.h5')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    inverted = 255 - resized
    normalized = inverted.astype('float32') / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

def predict_digit(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    return digit

def webcam_capture():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Webcam', frame)
        
        digit = predict_digit(frame)
        print(f'Predicted digit: {digit}')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model = load_model('mnist_digit_recognition.h5')
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.predict_button = tk.Button(root, text="Predict Digit", command=self.predict)
        self.predict_button.pack()
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        radius = 10
        self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='black')
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='black', outline='black')

    def predict(self):
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img = np.array(img)
        img = 255 - img  
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        print(f'Predicted digit: {digit}')
        
        messagebox.showinfo("Prediction", f"The digit is: {digit}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
