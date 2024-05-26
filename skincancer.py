import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageTk


# Define labels
Labels = ['Benign', 'Malignant']

# Load the trained model
module_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, FV_SIZE = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(MODULE_HANDLE, input_shape=(224, 224, 3))

model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(Labels), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load model weights
model.load_weights('D:\Cancer Detector\cancer_model.h5')  

# Define preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

# Define prediction function
def predict_skin_cancer(image):
    preprocessed_img = preprocess_image(image)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    prediction = model.predict(preprocessed_img)[0]
    max_prob_index = np.argmax(prediction)
    
    skin_cancer_types = {
        0: "Basal Cell Carcinoma",
        1: "Benign Keratosis",
        2: "Dermatofibroma",
        3: "Melanoma",
        4: "Melanocytic Nevi",
        5: "Vascular Lesions"
    }
    
    predicted_class = Labels[max_prob_index]
    skin_cancer_type = skin_cancer_types[max_prob_index]
    confidence = round(prediction[max_prob_index] * 100, 2)
    
    return predicted_class, skin_cancer_type, confidence

# Handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        predicted_class, skin_cancer_type, confidence = predict_skin_cancer(image)
        display_image(image)
        display_prediction(predicted_class, skin_cancer_type, confidence)

# Display the uploaded image
def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    aspect_ratio = width / height
    new_height = 350
    new_width = int(new_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))
    photo = ImageTk.PhotoImage(image=Image.fromarray(image))
    image_label.config(image=photo)
    image_label.image = photo

# Display the prediction
def display_prediction(predicted_class, skin_cancer_type, confidence):
    explanation = {
        "Basal Cell Carcinoma": "Basal Cell Carcinoma is the most common type of skin cancer. It often appears as a pearly or waxy bump, or a flat, flesh-colored or brown scar-like lesion.",
        "Benign Keratosis": "Benign Keratosis, also known as seborrheic keratosis, is a non-cancerous skin growth that appears as a brown, black or pale growth on the face, chest, shoulders or back.",
        "Dermatofibroma": "Dermatofibroma is a harmless skin growth that usually appears as a small, red or brown bump. It's often found on the legs and is caused by an accumulation of fibroblasts.",
        "Melanoma": "Melanoma is the deadliest form of skin cancer that develops from melanocytes, the cells that produce melanin. It often appears as an unusual mole or dark spot on the skin.",
        "Melanocytic Nevi": "Melanocytic Nevi, commonly known as moles, are benign growths that appear as small, dark spots or raised bumps on the skin. They're usually harmless but should be monitored for changes.",
        "Vascular Lesions": "Vascular Lesions include various types of blood vessel abnormalities such as hemangiomas and port-wine stains. They may appear as red or purple patches or bumps on the skin.",
    }

    # Display the predicted class, type of skin cancer, confidence, and explanation
    prediction_label.config(text=f"Predicted class: {predicted_class}\nType of Skin Cancer: {skin_cancer_type}\nConfidence: {confidence}%\n\n{explanation[skin_cancer_type]}")


# Create the main window
root = tk.Tk()
root.title("Skin Cancer Detector")
root.configure(bg='white')

# Set window size to screen resolution
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (screen_width, screen_height))

# Set the title with specific font, size, and color
title_label = tk.Label(root, text="Skin Cancer Detector".upper(), font=("Sylfaen", 37, "bold"), fg="black", bg='white')
title_label.pack(pady=20)

# Create labels for image and prediction
image_label = tk.Label(root)
image_label.pack(pady=10)
prediction_label = tk.Label(root, text="", font=("Arial", 12), bg='white')
prediction_label.pack(pady=10)

# Create a button for image upload
upload_button = tk.Button(root, text="Upload Image".upper(), command=upload_image, font=("Times New Roman", 15), fg="Blue", width=15, height=2)
upload_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
