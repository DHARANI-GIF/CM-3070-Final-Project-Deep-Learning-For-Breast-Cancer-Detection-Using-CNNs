# CM-3070-Final-Project-Deep-Learning-For-Breast-Cancer-Detection-Using-CNNs
# CM-3070-Final-Project: Deep Learning for Breast Cancer Detection Using CNNs

## Overview of prototype application

The **Breast Cancer Diagnostic Assistant** is a web-based application designed to assist radiologists in detecting breast cancer from mammogram images. The app integrates deep learning models using Convolutional Neural Networks (CNNs) trained on the CBIS-DDSM dataset for multi-class classification. The tool classifies lesions into four categories:

- **Benign Mass**
- **Malignant Mass**
- **Benign Calcification**
- **Malignant Calcification**

This project utilizes Streamlit for the user interface and TensorFlow for the deep learning model. The assistant provides real-time predictions, generates PDF reports, and includes features for secure user authentication and audit logging.

## Key Features

- **User Authentication**: Password-protected login to ensure controlled access.
- **Patient Metadata Collection**: Input patient details such as ID, age, and medical history.
- **Image Upload and Preprocessing**: Upload mammogram images (JPEG/PNG), which are preprocessed before classification.
- **Real-time Prediction**: Predictions for lesion types with confidence scores and priority levels.
- **Result Visualization**: Display images with predicted class and priority (color-coded).
- **PDF Report Generation**: Automatic generation of PDF reports including images and diagnostic results.
- **Audit Logging**: Captures and stores a detailed log of all actions for traceability.

## Installation

### Prerequisites

To run this project, make sure you have the following installed:

- Python 3.7+
- TensorFlow 2.x
- Streamlit 1.x
- FPDF (for PDF report generation)
- Other Python dependencies listed in `requirements.txt`

### Clone the Repository

```bash
git clone https://github.com/your-username/DHARANI-GIF.git
cd DHARANI-GIF/FYP Project App
```

### Run the Streamlit App

To run the application, use the following command:
```bash

streamlit run app.py
```

This will launch the app in your default web browser.

## Usage
### User Authentication

Upon accessing the app, you'll be prompted to log in using your credentials.

Enter the required password(radiology123). Incorrect attempts will be flagged.

### Patient Metadata Collection

After logging in, the radiologist can enter patient details including ID, age, sex, family history, etc.

All fields marked with * are mandatory.

### Image Upload and Preprocessing

Radiologists can upload mammogram images (JPEG or PNG).

The system will automatically preprocess the images for compatibility with the model.

Predictions and Result Visualization

Once the image is processed, the model will predict the class of the lesion.

The results, along with the confidence level, are displayed alongside the images.

Color-coded annotations indicate the priority of the cases (red for high priority, green for benign, etc.).

### PDF Report Generation

A PDF report is generated automatically, which includes the patient's metadata, image predictions, and confidence scores.

The report can be downloaded directly from the app.

### Audit Logging

All actions taken within the app are logged for traceability.

These logs are stored in a CSV format.

## Models

The project includes the following models:

**CM_3070_Binary(Mass_vs_Calcification).ipynb**: CNN built from scratch to classify Mass vs Calcification.

**CM_3070_Multi_Class.ipynb**: CNN built from scratch for 4-class classification.

**CM_3070_Binary(Benign_vs_Malignant)**: CNN built from scratch for benign vs malignant classification.

**CM_3070_Binary_VGG16.ipynb**: VGG16 model with feature extraction for 2-class classification.

**CM_3070_Multi_Class_VGG16.ipynb**: VGG16 model with feature extraction for 4-class classification.

**CM_3070_Composite_Multi_Class.ipynb**: Two parallel CNN models for decomposing the 4-class task.

**CM_3070_Ensemble_Binary.ipynb**: Ensemble model of various CNNs for 2-class classification.

**CM_3070_Ensemble_Multi_Class.ipynb**: Ensemble model of various CNNs for 4-class classification.

For model details, refer to the corresponding .ipynb files.

## License

This project is licensed under the MIT License - see the LICENSE
 file for details.

## Acknowledgments

CBIS-DDSM dataset for breast cancer mammogram images.

TensorFlow and Keras for deep learning model development.

Streamlit for building the web-based user interface.

FPDF for PDF report generation.
