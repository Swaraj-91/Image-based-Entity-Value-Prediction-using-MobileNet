import os
import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Fix for truncated image files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load pretrained MobileNet model for feature extraction
mobilenet_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')

def download_image(image_url):
    """Download an image from a URL."""
    try:
        if image_url:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
    except IOError as e:
        print(f"Error opening image: {e}")
    return None

def process_image(img):
    """Process an image to extract features using MobileNet."""
    if img is None:
        return np.zeros((1024,))
    try:
        img = img.convert("RGB")  # Ensure image is in RGB format
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = mobilenet_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image: {e}")
        return np.zeros((1024,))

def parse_entity_value(entity_value):
    try:
        match = re.match(r'^(\d+\.?\d*)(\s?\w+)?$', entity_value.strip())
        if match:
            return float(match.group(1))  # Extract the numeric part and convert to float
    except (ValueError, IndexError):
        pass
    return None

def process_row(row, idx, total_images):
    """Process a single row to extract features and value."""
    img = download_image(row.get('image_link', ''))
    features = process_image(img)
    entity_value = row.get('entity_value', '')
    
    value = parse_entity_value(entity_value)
    if value is None:
        print(f"Invalid entity value at row {idx}: {entity_value}")
        value = 0  # Defaulting to zero if the value cannot be parsed
    print(f"Processing image {idx + 1}/{total_images}")
    return features, value

def train_model(train_data):
    """Train the model using the provided training data."""
    X, y = [], []
    total_images = len(train_data)
    
    for idx, row in train_data.iterrows():
        try:
            features, value = process_row(row, idx, total_images)
            X.append(features)
            y.append(value)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    X = np.array(X)
    y = np.array(y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X, y)
    return pipeline

def predictor(model, image_link, entity_name):
    """Predict the value for a given image link and entity name."""
    img = download_image(image_link)
    features = process_image(img).reshape(1, -1)
    
    try:
        predicted_value = model.predict(features)[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        return ""  # Return empty string if there's an issue with prediction

    entity_unit_map = {
        "width": "centimetre",
        "height": "centimetre",
        "depth": "centimetre",
        "item_weight": "gram",
        "maximum_weight_recommendation": "gram",
        "voltage": "volt",
        "wattage": "watt",
        "item_volume": "litre"
    }
    
    unit = entity_unit_map.get(entity_name, "")
    return f"{predicted_value:.2f} {unit}" if unit else f"{predicted_value:.2f}"

def evaluate_f1_score(test_data):
    """Evaluate F1 score based on test data."""
    y_true, y_pred = [], []
    
    for idx, row in test_data.iterrows():
        ground_truth = row.get('ground_truth', '').strip()
        prediction = row.get('prediction', '').strip()

        if ground_truth and prediction:
            y_true.append(ground_truth)
            y_pred.append(prediction)

    y_true = [1 if gt == 'True' else 0 for gt in y_true]
    y_pred = [1 if pred == 'True' else 0 for pred in y_pred]

    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    
    return precision, recall, f1

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    
    train_file = os.path.join(DATASET_FOLDER, 'train.csv')
    test_file = os.path.join(DATASET_FOLDER, 'test.csv')
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
    
        # Train the model on the training data
        model = train_model(train)
        
        # Make predictions on the test set
        test['prediction'] = test.apply(
            lambda row: predictor(model, row.get('image_link', ''), row.get('entity_name', '')) if row.get('image_link', '') else "", axis=1)
        
        # Evaluate F1 score
        precision, recall, f1 = evaluate_f1_score(test)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        
        # Save the predictions in the required format
        output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
        test[['index', 'prediction']].to_csv(output_filename, index=False)
        
        print(f"Predictions saved to {output_filename}")
    else:
        print("Train or test data file not found.")
