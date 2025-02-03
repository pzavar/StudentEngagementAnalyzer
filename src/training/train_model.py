"""
Model training script for emotion classification using MobileNetV2.
This script assumes the emotion dataset is loaded into a pandas DataFrame 'df'
with columns for image paths and emotion labels.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

def create_model(input_shape: tuple = (224, 224, 3)) -> Model:
    """
    Create and return the neural network model using MobileNetV2.

    Args:
        input_shape (tuple): Input shape for the model

    Returns:
        Model: Compiled Keras model
    """
    # Use MobileNetV2 as base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze base model layers
    base_model.trainable = False

    # Add classification head
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(7, activation='softmax')(x)  # 7 emotion classes

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def preprocess_data(df: pd.DataFrame, input_shape: tuple = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the image data and prepare labels.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels
        input_shape (tuple): Target size for image resizing

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed images and one-hot encoded labels
    """
    images = []
    labels = []

    for idx, row in df.iterrows():
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            row['image_path'],
            target_size=input_shape
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        images.append(img_array)
        labels.append(row['emotion'])

    # Convert to numpy arrays
    X = np.array(images)
    y = tf.keras.utils.to_categorical(labels)

    return X, y

def train_model(
    df: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2
) -> Tuple[Model, Dict[str, Any]]:
    """
    Train the emotion classification model with advanced training features.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation

    Returns:
        Tuple[Model, Dict[str, Any]]: Trained model and training history
    """
    # Preprocess data
    X, y = preprocess_data(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=validation_split, 
        random_state=42,
        stratify=y
    )

    # Create and compile model
    model = create_model()

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history.history

def plot_training_results(history: Dict[str, Any], save_path: str = 'training_results') -> None:
    """
    Plot detailed training metrics using seaborn.

    Args:
        history (Dict[str, Any]): Training history from model.fit()
        save_path (str): Base path for saving visualization files
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)

    # Plot accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=history['accuracy'], label='Training Accuracy', ax=ax1)
    sns.lineplot(data=history['val_accuracy'], label='Validation Accuracy', ax=ax1)
    ax1.set_title('Model Accuracy Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')

    # Plot loss
    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=history['loss'], label='Training Loss', ax=ax2)
    sns.lineplot(data=history['val_loss'], label='Validation Loss', ax=ax2)
    ax2.set_title('Model Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    # Plot learning rate
    if 'lr' in history:
        ax3 = fig.add_subplot(gs[1, 0])
        sns.lineplot(data=history['lr'], ax=ax3)
        ax3.set_title('Learning Rate Over Time')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')

    # Add training summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary_text = (
        f"Training Summary:\n"
        f"Best Validation Accuracy: {max(history['val_accuracy']):.4f}\n"
        f"Final Training Accuracy: {history['accuracy'][-1]:.4f}\n"
        f"Best Validation Loss: {min(history['val_loss']):.4f}\n"
        f"Final Training Loss: {history['loss'][-1]:.4f}"
    )
    ax4.text(0.1, 0.5, summary_text, fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_path}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot confusion matrix (if available)
    if 'confusion_matrix' in history:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            history['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='YlOrRd'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{save_path}_confusion.png', dpi=300, bbox_inches='tight')
        plt.close()

def main(df: pd.DataFrame):
    """
    Main training function with enhanced progress tracking.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and emotion labels
    """
    print("Starting model training process...")

    # Train model with progress tracking
    model, history = train_model(df)

    # Generate and save detailed visualizations
    print("Generating training visualizations...")
    plot_training_results(history)

    print("Training process completed successfully!")
    print("- Model saved as 'best_model.h5'")
    print("- Training visualizations saved as 'training_results_*.png'")

if __name__ == "__main__":
    # Example usage (assuming df is loaded)
    # main(df)
    pass