# Step 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Step 2: Load Dataset
def load_dataset(csv_path):
    data = pd.read_csv("Face_data.csv")
    if 'Label' not in data.columns:
        raise ValueError("CSV file must contain 'Label' column.")
    X = data.drop('Label', axis=1)
    y = data['Label']
    print(f"Dataset shape: {data.shape}")
    print(f"Unique classes: {len(np.unique(y))}")
    print(data.head())
    return X, y
csv_path = r"Face_data.csv" # Use your path here
X, y = load_dataset(r"Face_data.csv")
# Step 3: Scale features for traditional ML models
def preprocess_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled.")
    return X_scaled, scaler
X_scaled, scaler = preprocess_features(X)
# Step 4: Split dataset for traditional models
def split_dataset(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split_dataset(X_scaled, y)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print("Training data shape:", X_train_smote.shape)
print("Training label distribution:", dict(pd.Series(y_train_smote).value_counts()))

# Step 5: Train and evaluate traditional ML models
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    accuracies = {}
    reports = {}
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        reports[name] = classification_report(y_test, y_pred)
        trained_models[name] = model
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Classification Report:\n{reports[name]}")
        plot_confusion_matrix(y_test, y_pred, name)
    return accuracies, reports, trained_models
accuracies, reports, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv(r"Face_data.csv")

# Define features and label
X = df.drop("Label", axis=1)
y = df["Label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    accuracies = {}
    reports = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        reports[name] = classification_report(y_test, y_pred)
        trained_models[name] = model
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Classification Report:\n{reports[name]}")
        plot_confusion_matrix(y_test, y_pred, name)

    return accuracies, reports, trained_models

# Train all models
accuracies, reports, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Get best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} with accuracy: {accuracies[best_model_name]*100:.2f}%")

# Save best model
with open("best_face_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Best model saved as 'best_face_model.pkl'")

# Import necessary packages
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Assume X and y are already defined (X = features, y = target)
# Example: X = df.drop('target', axis=1); y = df['target']

# Step 1: Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 3: Continue with model training
print("Before SMOTE:", X_train.shape, y_train.value_counts())
print("After SMOTE:", X_train_smote.shape, y_train_smote.value_counts())


accuracies, reports, models = train_and_evaluate_models(X_train_smote, X_test, y_train_smote, y_test)

# Step 6: Accuracy comparison bar plot
def plot_accuracy_comparison(accuracies):
    plt.figure(figsize=(10, 6))
    names = list(accuracies.keys())
    values = [accuracies[name]*100 for name in names]
    sns.barplot(x=values, y=names, orient='h', palette='viridis')
    plt.xlabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.xlim(0, 100)
    for i, v in enumerate(values):
        plt.text(v + 1, i, f"{v:.2f}%", va='center')
    plt.show()
plot_accuracy_comparison(accuracies)
# Step 7: Preprocess data for CNN - reshape and normalize
def preprocess_for_cnn(X):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    desired_size = 144  # 12x12 image
    if n_features < desired_size:
        pad_width = desired_size - n_features
        X_padded = np.hstack([X, np.zeros((n_samples, pad_width))])
    else:
        X_padded = X.iloc[:, :desired_size].values
    X_images = X_padded.reshape((n_samples, 12, 12, 1)).astype('float32') / 255.0
    print(f"Reshaped to CNN input shape: {X_images.shape}")
    return X_images
X_images = preprocess_for_cnn(X)
# Step 8: Split dataset for CNN training and testing
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_images, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train_img.shape[0]}, Test samples: {X_test_img.shape[0]}")
# Step 9: CNN model creation
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
cnn_model = create_cnn_model(X_train_img.shape[1:])
cnn_model.summary()
# Step 10: Train CNN model with augmentation and callbacks
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
train_generator = datagen.flow(X_train_img, y_train_img, batch_size=16)
steps_per_epoch = len(X_train_img) // 16
history = cnn_model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_test_img, y_test_img),
    callbacks=[early_stop, reduce_lr],
    verbose=2
)
# Evaluate CNN
loss, accuracy = cnn_model.evaluate(X_test_img, y_test_img, verbose=0)
print(f"\nCNN Model Test Accuracy: {accuracy*100:.2f}%")
# Predict and report
y_pred_prob = cnn_model.predict(X_test_img)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
print(f"\nClassification Report (CNN):\n{classification_report(y_test_img, y_pred)}")
plot_confusion_matrix(y_test_img, y_pred, "CNN") 
best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]
print(f"\nBest traditional model: {best_model_name} with accuracy: {best_accuracy*100:.2f}%")
print(f"\nSample predictions using {best_model_name} model on first 3 test samples:")
sample_preds = models[best_model_name].predict(X_test[:3])
for i, pred in enumerate(sample_preds, 1):
    print(f"Sample {i} predicted label: {pred}")


import tensorflow as tf
print("TensorFlow version:", tf.__version__)
