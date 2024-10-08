# README: Image Feature Extraction and Regression Model Pipeline

## Overview

This project focuses on developing a machine learning pipeline that:
1. Downloads images from URLs.
2. Extracts features using a pre-trained **MobileNet** model.
3. Trains a **Random Forest Regressor** to predict numerical values associated with entities (e.g., weight, height) from the image features.
4. Provides a prediction pipeline to estimate the value of specific entities for new images.
5. Evaluates the model performance using classification metrics like **Precision**, **Recall**, and **F1 Score** for validation.

The pipeline also handles image processing errors, data pre-processing, and prediction generation in an efficient and automated manner.

---

## Project Components

### 1. **MobileNet Feature Extraction**
The pre-trained **MobileNet** model, provided by TensorFlow, is used for image feature extraction. The model is loaded without its top classification layer and generates a 1024-dimensional feature vector from input images.

### 2. **Image Processing**
Images are downloaded from the provided URLs using the \
equests\ library. After downloading, each image is resized to \224x224\ and converted to a 3-channel RGB format. The \PIL\ library handles any potential image corruption issues (\LOAD_TRUNCATED_IMAGES = True\).

### 3. **Data Parsing**
Entity values, associated with each image (e.g., height, weight), are parsed from text using regular expressions to extract numerical values. Default values are provided for any unprocessable entries to ensure smooth model training.

### 4. **Model Training**
The regression model uses a **Random Forest Regressor** to predict continuous values (e.g., dimensions, weights) based on image features. The feature extraction and regression pipeline is composed of:
- **StandardScaler**: Standardizes the feature vectors.
- **RandomForestRegressor**: Trains a regression model with 100 estimators.

### 5. **Prediction**
A prediction pipeline estimates the entity value for new images. It also maps the predicted values to their respective units (e.g., cm, grams) based on the entity name (width, height, weight, etc.).

### 6. **Evaluation**
The **F1 Score** and associated metrics (Precision, Recall) are calculated for the test data to evaluate classification performance. This assumes the ground truth values are boolean-like, allowing the conversion into binary format for scoring.

### 7. **File Handling**
The pipeline reads the input data from \	rain.csv\ and \	est.csv\ files, makes predictions, and saves the results into a \	est_out.csv\ file.

---

## Directory Structure

\\\
project_directory/
│
├── dataset/
│   ├── train.csv         # Training data with images and entity values
│   ├── test.csv          # Test data for predictions and evaluations
│   └── test_out.csv      # Output file with predictions
│
├── code.py        # Main code for training, prediction, and evaluation
├── README.md             # Documentation
└── requirements.txt      # Python dependencies
\\\

---

## How to Run the Project

1. **Clone the Repository**:
   \\\ash
   git clone <repository_url>
   cd project_directory
   \\\

2. **Install Dependencies**:
   Ensure Python 3.6+ is installed. Then run:
   \\\ash
   pip install -r requirements.txt
   \\\

3. **Prepare the Dataset**:
   Place the \	rain.csv\ and \	est.csv\ files inside the \dataset/\ directory. These files should contain the following columns:
   - \image_link\: URL of the image.
   - \entity_value\: The target value (e.g., weight, height, etc.) for the image.
   - \entity_name\: The name of the entity to predict.
   - \ground_truth\: (for test data) The true values to compare against the predictions.

4. **Run the Script**:
   Run the \main_script.py\ to train the model and make predictions:
   \\\ash
   python main_script.py
   \\\

5. **Check Results**:
   After running the script, the predictions will be saved in the \dataset/test_out.csv\ file. The script will also print the **Precision**, **Recall**, and **F1 Score** metrics.

---

## Input and Output Details

### Input Format
- **Train.csv**: Contains the image URL and entity values.
- **Test.csv**: Contains the image URL, entity names, and optional ground truth for evaluation.

| index | image_link                  | entity_value | entity_name  | ground_truth |
|-------|-----------------------------|--------------|--------------|--------------|
| 0     | https://image-url.com/image1 | 45 kg        | item_weight  | True         |

### Output Format
- **Test_out.csv**: Contains the predictions for each test image, stored in the \dataset/test_out.csv\ file.

| index | prediction     |
|-------|----------------|
| 0     | 45.67 kilogram |

---

## Key Functions

### 1. **\download_image(image_url)\**
Downloads and opens an image from the provided URL. Handles request timeouts and invalid image links gracefully.

### 2. **\process_image(img)\**
Processes an image (resize, preprocess, and extract features) using **MobileNet**.

### 3. **\parse_entity_value(entity_value)\**
Parses and converts entity values from string format to float.

### 4. **\process_row(row, idx, total_images)\**
Processes a single data row: downloads the image, extracts features, parses the entity value, and logs progress.

### 5. **\	rain_model(train_data)\**
Trains the regression model using **RandomForestRegressor** and returns a pipeline.

### 6. **\predictor(model, image_link, entity_name)\**
Uses the trained model to predict the entity value for a given image.

### 7. **\evaluate_f1_score(test_data)\**
Evaluates the **Precision**, **Recall**, and **F1 Score** for the test set.

---

## Evaluation Metrics

The code evaluates the model using **Precision**, **Recall**, and **F1 Score** to assess the performance of binary predictions. These metrics are commonly used in classification tasks to balance between false positives and false negatives.

---

## Dependencies

This project requires the following Python packages:

- \	ensorflow\: For feature extraction using **MobileNet**.
- \scikit-learn\: For machine learning tasks (regression, evaluation metrics).
- \pandas\: For data manipulation.
- \
umpy\: For numerical computations.
- \Pillow\: For image processing.
- \
equests\: For downloading images.

Install all dependencies using:
\\\ash
pip install -r requirements.txt
\\\

---

## Future Improvements

- **Model Optimization**: Experiment with different regressors (e.g., XGBoost) for improved performance.
- **Error Handling**: Improve error handling during prediction for unseen or corrupted image data.
- **Additional Features**: Introduce more feature extraction models and compare their efficacy.
- **Scalability**: Optimize the pipeline to handle larger datasets and more complex model architectures.

---

## Contact

For any issues or inquiries, please contact the project maintainer at \<jagadeswaraj91@gmail.com>\.

