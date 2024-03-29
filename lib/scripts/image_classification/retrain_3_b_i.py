import os
import yaml
import time
import shutil

import tensorflow as tf
from collections import Counter

from tensorflow import keras
print("TensorFlow version is ", tf.__version__)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import lib.scripts.file_utils as fu


MOBILENET_V1 = 'mobilenet_v1'
MOBILENET_V2 = 'mobilenet_v2'
MOBILENET_V3 = 'mobilenet_v3'
INCEPTION_V3 = 'inception_v3'


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join('..', '..', 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'retrain_3_b_i.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    train_dir = config["train_dir"]
    val_dir = config["val_dir"]
    train_sums_dir = config["training_summaries"]
    model_dir = config["model_dir"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    model_input_size = config["model_input_size"]
    alpha = config["alpha"]
    architecture = config["architecture"]
    fine_tune_at = config["fine_tune_at"]

    # create training_summaries dir if it doesn't exist
    fu.init_directory(directory=train_sums_dir)

    # copy config to output dir for book keeping
    shutil.copyfile(src=yaml_path, dst=os.path.join(model_dir, os.path.basename(yaml_path)))

    # get class directories
    class_image_paths = {}
    class_dirs = [d for d in os.listdir(train_dir)]

    # rescale all images by 1./255 and apply image augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                 brightness_range=[0.7, 1.3],
                                                                 zoom_range=[0.7, 1.3],
                                                                 horizontal_flip=True,
                                                                 rotation_range=30,
                                                                 width_shift_range=20,
                                                                 height_shift_range=20,
                                                                 fill_mode='reflect')

    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # Source directory for the training images
        target_size=(model_input_size, model_input_size),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

    # balance classes since they may be imbalanced
    print("class_indices: {}".format(train_generator.class_indices))
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
    print("class weights: {}".format(class_weights))

    # write training labels to file for later use
    with open(os.path.join(model_dir, 'train_labels.txt'), "w+") as f:
        for label in train_generator.class_indices:
            f.write("{}\n".format(label))

    # flow validation images in batches of 20 using test_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
        val_dir,  # Source directory for the validation images
        target_size=(model_input_size, model_input_size),
        batch_size=batch_size,
        class_mode='categorical')

    IMG_SHAPE = (model_input_size, model_input_size, 3)

    # Create the base model from a pre-trained model

    # MobileNet V2
    base_model = None

    if architecture == MOBILENET_V1:
        base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                                     include_top=False,
                                                     weights='imagenet',
                                                     alpha=alpha)
    elif architecture == MOBILENET_V2:
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet',
                                                       alpha=alpha)
    elif architecture == MOBILENET_V3:
        base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                            include_top=False,
                                                            weights='imagenet',
                                                            alpha=alpha)
    elif architecture == INCEPTION_V3:
        base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

    # setup tensorboard logging callback
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=train_sums_dir
    )

    # setup early stopping and model checkpoint callback to save model with lowest loss
    early_stop_min_loss_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=3,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True
    )

    # define training callbacks
    callbacks = [
        tensorboard_cb,
        early_stop_min_loss_cb
    ]

    ## Fine-tune last layer

    # let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # let's visualize layer names and layer indices
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # freeze model's base layers
    base_model.trainable = False

    # look at the base model architecture
    base_model.summary()

    # add new classification head
    model_k = tf.keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=len(class_dirs), activation='softmax')
    ])

    # compile the model
    model_k.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # get new model summary
    model_k.summary()

    # train the model
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size

    t1 = time.time()
    history = model_k.fit_generator(train_generator,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    workers=4,
                                    validation_data=validation_generator,
                                    validation_steps=validation_steps,
                                    class_weight=class_weights,
                                    callbacks=callbacks)
    t2 = time.time()
    print("Fine-tuning starting at last layer took: {}s".format(t2 - t1))

    # view train/val accuracy and loss
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(train_sums_dir, 'acc_loss_for_fine_tuned_last_layer.png'))
    # plt.show()

    # save intermediate model
    model_k.save(filepath=os.path.join(model_dir, 'retrained_model_last_layer.h5'))

    ## Fine-tune more layers

    # allow all layers to be trainable
    base_model.trainable = True

    # let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # compile the model
    model_k.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # get new model summary
    model_k.summary()

    # train the model
    t3 = time.time()
    history_fine = model_k.fit_generator(train_generator,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epochs+epochs, # epochs is understood as final epoch to train until reached, NOT a total number of epochs to train for
                                         workers=4,
                                         validation_data=validation_generator,
                                         validation_steps=validation_steps,
                                         callbacks=callbacks,
                                         initial_epoch=epochs)
    t4 = time.time()
    print("Fine-tuning starting at layer {} took: {}s".format(fine_tune_at, t2 - t1))

    # view train/val accuracy and loss
    acc += history_fine.history['categorical_accuracy']
    val_acc += history_fine.history['val_categorical_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.9, 1])
    plt.plot([epochs - 1, epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 0.2])
    plt.plot([epochs - 1, epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(train_sums_dir, 'acc_loss_for_fine_tuning_layers_starting_at_{}.png'.format(fine_tune_at)))
    # plt.show()

    # save intermediate model
    model_k.save(filepath=os.path.join(model_dir, 'retrained_model_starting_at_layer_{}.h5'.format(fine_tune_at)))