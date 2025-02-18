#Data is already augmented, so we will load the model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import kagglehub
from data_sourcing import download_data()


def train_nn():
    #Normalize the images
    train_normalize = ImageDataGenerator(
        rescale = 1./255,
        validation_split=0.2
    )

    corr_path = download_data()

    #Load and normalize the images from the path
    #Create the training set
    train_set = train_normalize.flow_from_directory(
        corr_path,
        target_size = (224,224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    #Create the validation set
    val_set = train_normalize.flow_from_directory(
        corr_path,
        target_size = (224,224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    #Loading the pre_trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    #Freeze the model weights
    base_model.trainable = False

    #Building a model on top of this
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_set.num_classes, activation='softmax')
    ])

    #Model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    #train the model
    training = model.fit(
        train_set,
        epochs=10,
        validation_data=val_set,
        verbose=1
    )

    #Prints the loss and accuracy
    val_loss, val_accuracy = model.evaluate(val_set)
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

    # Define the folder path
    folder_path = 'AIPI540_BananaTreeClassification/models/'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save the model in the specified folder
    model.save(os.path.join(folder_path, 'tree_health_classification_nn.keras'))


