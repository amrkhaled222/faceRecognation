
import pygame
import tkinter as tk
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageTk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pyttsx3
# Initialize pygame
pygame.init()

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
else:
    print("No GPU devices found.")# Define the path to your dataset directory

dataset_directory = 'Dataset'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to load and preprocess the images
def resize_with_padding(image, target_size=(64, 64)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    # Resize image
    resized_image = cv2.resize(image, (resized_w, resized_h))
    # Create a blank canvas with the target size
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    # Center the resized image on the canvas
    start_x = (target_size[1] - resized_w) // 2
    start_y = (target_size[0] - resized_h) // 2
    canvas[start_y:start_y+resized_h, start_x:start_x+resized_w] = resized_image
    return canvas

def load_images(dataset_directory, target_size=(64, 64)):
    image_data = []
    labels = []
    for label, person in enumerate(sorted(os.listdir(dataset_directory))):
        for file_name in os.listdir(os.path.join(dataset_directory, person)):
            image_path = os.path.join(dataset_directory, person, file_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for face detection
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                print(f"No face detected in image: {image_path}")
                continue
            # Assume the first detected face is the one to use
            x, y, w, h = faces[0]
            face = image[y:y + h, x:x + w]
            # Resize the face region
            image = cv2.resize(face, (64, 64))  # Resize the image to a fixed size
 
            image_data.append(image)
            labels.append(label)

    return np.array(image_data), np.array(labels)


def create_model_v2(input_shape, num_classes):
    model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
    ])

    lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)

    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
# Train the model

def train_model(train_images, train_labels, test_images, test_labels):
    # Normalize pixel values
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # Create and compile the model
    model = create_model_v2(input_shape=train_images[0].shape, num_classes=len(np.unique(train_labels)))

    # Early Stopping callback
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

    # Train the model
    history = model.fit(train_images, train_labels, epochs=50, batch_size=16, verbose=1)

    # Evaluate the model
    _, accuracy = model.evaluate(test_images, test_labels)

    # Predict probabilities for test images
    y_pred_probs = model.predict(test_images)

    # Calculate precision for each class
    precision = precision_score(test_labels, np.argmax(y_pred_probs, axis=1), average=None)

    return model, accuracy, y_pred_probs, precision

def evaluate_model(model, test_images, test_labels):
    # Normalize pixel values
    test_images = test_images / 255.0
    
    # Evaluate the model
    _, accuracy = model.evaluate(test_images, test_labels)
    
    # Predict probabilities for test images
    y_pred_probs = model.predict(test_images)
    
    # Calculate precision for each class
    precision = precision_score(test_labels, np.argmax(y_pred_probs, axis=1), average=None)
    
    # Print evaluation metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    for i in range(len(np.unique(train_labels))):
        fpr[i], tpr[i], _ = roc_curve(test_labels, y_pred_probs[:, i], pos_label=i)
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(test_labels))):
        plt.plot(fpr[i], tpr[i], label=f'Class {i}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Calculate and plot confusion matrix
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(test_labels, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def close_camera(cap, root, model, test_images, test_labels):
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    
    # Evaluate the model after closing the camera
    evaluate_model(model, test_images, test_labels)

def open_camera(model, test_images, test_labels):
    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Failed to open camera.")
        return False

    # Variables to track recognition status
    recognized = False

    # Set font properties for the recognized name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2

    # Create Tkinter window
    root = tk.Tk()
    root.title("Face Recognition")

    # Create a canvas to display the camera feed
    camera_canvas = tk.Canvas(root, width=640, height=480, bg="blue")
    camera_canvas.pack(pady=10)

    # Create close button
    close_button = tk.Button(root, text="Close Camera", command=lambda: close_camera(cap, root, model, test_images, test_labels), bg="blue", fg="white")
    close_button.pack()
    
    # Define the sound file path(correct sound )
    sound_file = 'C:/Users/Admin/Desktop/know_me/sounds/Correct_Sound.mp3'
    
    # # Function to play the correct sound
    # def play_sound():
    #     sound = pygame.mixer.Sound(sound_file)
    #     sound.play()
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

# Function to speak the name of the recognized person
    def speak_name(name):
       engine.setProperty('rate', 120)  
       engine.say(name)
       engine.runAndWait()

    # Function to update camera feed
  
    def update_camera_feed():
        nonlocal recognized
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Failed to capture frame.")
            return

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            # Draw a green rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0,255), 2)

            # Crop the face region
            face = frame[y:y + h, x:x + w]
            # Resize and preprocess the face image
            face = cv2.resize(face, (64, 64))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) / 255.0
            face = np.expand_dims(face, axis=0)
            # Perform inference using the CNN model
            prediction = model.predict(face)
            predicted_label = np.argmax(prediction)
            # Get the name of the person based on the predicted label
            person_name = os.listdir(dataset_directory)[predicted_label]
            # Display the name of the person with green color and different style if recognized
            if np.max(prediction) > 0.5:  # Adjust recognition threshold as needed
                cv2.putText(frame, person_name, (x, y + h + 20), font, font_scale, (0, 255, 0), font_thickness,
                            cv2.LINE_AA)
                recognized = True
                speak_name(person_name)  # Play sound when recognized
            else:
                cv2.putText(frame, person_name, (x, y + h + 20), font, font_scale, (0, 0, 255), font_thickness,
                            cv2.LINE_AA)
                recognized = False

        # Display the frame on the Tkinter canvas
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        camera_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        camera_canvas.image = img

        # Check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_camera(cap, root, model, test_images, test_labels)
            return

        # Update the camera feed
        camera_canvas.after(10, update_camera_feed)
    
    # Start updating camera feed after defining the function
    update_camera_feed()

    # Run the Tkinter event loop
    root.mainloop()

# Define the number of folds
k = 10
# Load and preprocess the images
images, labels = load_images(dataset_directory)

# Initialize the StratifiedKFold object
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize lists to store ROC AUC values and precisions for each fold
roc_auc_values = []
precisions = []
accuracies = []
# Iterate over each fold
fold_index = 0
for train_indices, test_indices in kf.split(images, labels):
    print(f"Fold {fold_index + 1}:")
    
    # Get the images and labels for the current fold
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    print(f"Train size: {len(train_images)}, Test size: {len(test_images)}")
    
    # Train the model and get accuracy, predictions, and precision
    model, accuracy, y_pred_probs, precision = train_model(train_images, train_labels, test_images, test_labels)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(to_categorical(test_labels), y_pred_probs, multi_class='ovr')
    
    # Print metrics
    print(f'Accuracy: {accuracy}')
    print(f'ROC AUC: {roc_auc}')
    print(f'Precision: {precision}')
    
    # Append ROC AUC values and precisions to lists
    roc_auc_values.append(roc_auc)
    precisions.append(precision)
    accuracies.append(accuracy)
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    for i in range(len(np.unique(train_labels))):
        fpr[i], tpr[i], _ = roc_curve(test_labels, y_pred_probs[:, i], pos_label=i)
    
    
   
   
    
    fold_index += 1
# Print overall accuracy
print(f'Overall Accuracy: {np.mean(accuracies)}')
recognized = open_camera(model, test_images, test_labels)

    
# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(len(roc_auc_values)):
    plt.plot(fpr[i], tpr[i], label=f'Fold {i+1} (AUC = {roc_auc_values[i]:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Calculate and plot confusion matrix
if recognized:
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(test_labels, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
