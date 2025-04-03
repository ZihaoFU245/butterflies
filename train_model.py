import tensorflow as tf
from keras import layers, models, utils
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import cv2
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gc

# Temporarily disable mixed precision which can cause training issues
# mixed_precision = tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy('float32')  # Use standard precision for now

# Set GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def load_data(data_dir, img_size=(224, 224), max_samples=None):
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # Optionally limit number of samples for quicker development cycles
    if max_samples and max_samples < len(df):
        df = df.sample(max_samples, random_state=42)
    
    num_samples = len(df)
    print(f"Loading {num_samples} images...")

    # Process images in batches to save memory
    batch_size = 500
    images = []
    labels = []
    
    for i in range(0, num_samples, batch_size):
        batch_df = df.iloc[i:min(i+batch_size, num_samples)]
        
        def process_row(index, row):
            img_path = os.path.join(data_dir, "train", row["filename"])
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Unable to load image at {img_path}")
                return None, None

            # Convert BGR to RGB, resize, and normalize in one step
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size).astype('float32') / 255.0

            return img, row["label"]

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda x: process_row(*x), batch_df.iterrows()))

        # Collect processed images and labels
        for img, label in results:
            if img is not None:
                images.append(img)
                labels.append(label)
        
        print(f"Processed {min(i+batch_size, num_samples)}/{num_samples} images")
    
    # Convert to numpy arrays
    images = np.array(images, dtype='float32')
    
    # Ensure proper normalization by explicitly checking min/max
    if images.shape[0] > 0:
        print(f"Image min value: {images.min()}, max value: {images.max()}")
        
        # Ensure proper normalization if needed
        if images.max() > 1.0:
            images = images / 255.0
            print(f"Normalized image min: {images.min()}, max: {images.max()}")
    
    # Convert labels to numeric and one-hot encode
    label_dict = {label: idx for idx, label in enumerate(np.unique(labels))}
    num_label = [label_dict[label] for label in labels]
    labels = utils.to_categorical(num_label)

    return images, labels, label_dict

def create_CNN(input_shape, n_classes):
    model = models.Sequential([
        # First Convolutional Block with fewer filters to start
        layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Reduced dropout

        # Second Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),  # Reduced from 256
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Global Average Pooling
        layers.GlobalAveragePooling2D(),

        # Fully Connected Layers
        layers.Dense(256, activation='relu'),  # Reduced from 512
        layers.Dropout(0.4),  # Reduced dropout
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # Updated path for Linux environment
    data_dir = "/mnt/d/MyProject/Datasets/Butterflies"

    # Load data - optionally limit samples during development
    img, lab, label_dict = load_data(data_dir)
    
    # Free memory
    gc.collect()
    tf.keras.backend.clear_session()  # Clear TF session to free memory

    print(f"Images shape: {img.shape}")
    print(f"Labels shape: {lab.shape}")
    print(f"Number of classes: {len(label_dict)}")

    # Define model
    input_shape = (224, 224, 3)
    n_classes = len(label_dict)
    model = create_CNN(input_shape, n_classes)
    
    # Add mixed precision output layer to ensure correct output dtype
    model.add(layers.Activation('softmax', dtype='float32'))
    
    # Display model summary
    model.summary()

    # Use a smaller learning rate and standard optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Reduced learning rate
    
    # Add gradient clipping to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005, 
        clipnorm=1.0  # Add gradient clipping
    )
    
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )



    # Improve callbacks with better patience settings
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=20,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", 
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # More aggressive reduction
            patience=7,  # Increased patience
            min_lr=0.000001,
            verbose=1
        )
    ]

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(img, lab, test_size=0.2, random_state=42)
    
    # Free memory
    del img, lab
    gc.collect()

    # Determine optimal batch size based on your GPU - experiment with this value
    # Start with 64 and increase until you hit memory limits
    # Reduce batch size to avoid OOM errors
    BATCH_SIZE = 8  # Even smaller batch size for stable training

    # Apply data augmentation during training with optimized pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=min(1024, len(X_train)))
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y), 
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Examine a few samples to make sure data looks correct
    print("\nChecking training data...")
    for i in range(min(3, len(X_train))):
        print(f"Sample {i} - Shape: {X_train[i].shape}, " +
              f"Min: {X_train[i].min():.4f}, Max: {X_train[i].max():.4f}, " +
              f"Label: {np.argmax(y_train[i])}")

    # Train the model with more epochs to ensure sufficient training time
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=150,  # Increased epochs
        callbacks=callbacks,
        verbose=2  # More compact output
    )
    
    # save with a version number
    model.save("Butterfly_optimized.keras")

    # Free memory before evaluation
    gc.collect()
    
    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Make prediction in batches to avoid OOM during inference
    def predict_in_batches(model, x, batch_size=16):
        num_samples = len(x)
        predictions = np.zeros((num_samples, model.output_shape[1]))
        
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            batch_x = x[i:end]
            batch_pred = model.predict(batch_x, verbose=0)
            predictions[i:end] = batch_pred
        
        return predictions
    
    # Generate predictions for confusion matrix in batches
    print("Generating predictions for confusion matrix...")
    y_pred = predict_in_batches(model, X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    
    # Get class names from label dictionary
    class_names = [k for k, v in sorted([(k, v) for k, v in label_dict.items()], key=lambda x: x[1])]
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()



