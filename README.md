# Fruit Ripeness Prediction System

## Overview
The Fruit Ripeness Prediction System is an end-to-end computer vision application that classifies fruit images into **Fresh**, **Okay (Unripe)**, or **Avoid (Rotten)** categories.  
The project demonstrates the complete machine learning lifecycle — from data preprocessing and model training to deployment and real-time inference via a web interface.

---

## Key Features
- Image-based fruit ripeness classification
- Transfer learning using a pre-trained ResNet18 model
- Flask REST API for model inference
- Web-based frontend for image upload and camera capture
- Confidence score returned with every prediction
- Clean and modular project structure

---

## Dataset
- Source: Kaggle Fruit Ripeness Dataset
- link - https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten
- Total images: ~20,000
- Classes: Fresh, Okay (Unripe), Avoid (Rotten)
- Data split:
  - Training set
  - Validation set
  - Test set (held out for final evaluation)

---

## Model Architecture
- Backbone: ResNet18 (pre-trained on ImageNet)
- Strategy: Transfer learning
- Frozen convolutional layers
- Custom fully connected classifier (3 classes)

---

## Training Details
- Framework: PyTorch
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Input size: 224 × 224 RGB images
- Validation Accuracy: ~98.5%
- No significant overfitting observed

---

## System Architecture

Frontend (HTML, CSS, JavaScript)  
→ Flask REST API  
→ PyTorch Model (ResNet18)  
→ Prediction + Confidence Score

---

## How It Works
1. User uploads an image or captures a frame using the camera
2. Image is preprocessed (resize, normalization)
3. Model performs inference using the trained CNN
4. Flask API returns predicted label and confidence
5. Frontend displays the result in real time

---

## Technologies Used
- Python
- PyTorch
- Torchvision
- Flask
- HTML, CSS, JavaScript
- REST APIs

---

## Future Improvements
- Multi-fruit detection and localization
- Video stream-based inference
- Cloud deployment
- Mobile application integration

---

## Author
**Jatin Sharma**  
B.Tech CSE 
