import os
import yaml

import tensorflow as tf

from tensorflow import keras
print("TensorFlow version is ", tf.__version__)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import lib.file_utils as fu


MOBILENET_V2 = 'mobilenet_v2'

SHUFFLE_BUFFER = 2048


def _parse_function(example, tgt_size=None):
    # define your tfrecord again. Remember that you saved your image as a string.
    features = {'image/encoded': tf.FixedLenFeature([], tf.string), # tf.string = bytestring (not text string)
                "image/class/label": tf.FixedLenFeature([], tf.int64), # shape [] means scalar
                "image/height": tf.FixedLenFeature([], tf.int64),
                "image/width": tf.FixedLenFeature([], tf.int64)}

    # load one example
    parsed_example = tf.parse_single_example(example, features)

    # turn your saved image string into an array
    parsed_example['image/encoded'] = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)

    # resize image to target shape
    parsed_example['image/encoded'] = tf.image.resize(images=parsed_example['image/encoded'],
                                                      size=[tgt_size, tgt_size],
                                                      method=tf.image.ResizeMethod.BILINEAR)

    parsed_example['image/encoded'] = parsed_example['image/encoded'] / 255.0

    return parsed_example['image/encoded'], parsed_example["image/class/label"]


def create_dataset(filepaths=[], batch_size=None, num_classes=None, tgt_size=None):
    dataset = tf.data.TFRecordDataset(filepaths)

    # maps the parser on every filepath in the array. Can set the number of parallel loaders here
    dataset = dataset.map(lambda x: _parse_function(x, tgt_size=tgt_size), num_parallel_calls=8)

    # this dataset will go on forever
    dataset = dataset.repeat()

    # set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # set the batchsize
    dataset = dataset.batch(batch_size)

    # create an iterator
    iterator = dataset.make_one_shot_iterator()

    # create tf representation of the iterator
    image, label = iterator.get_next()

    # create a one hot array for labels
    label = tf.one_hot(label, num_classes)

    return image, label


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, '..', 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'retrain_3_a_i_tfr.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    data_dir = config["data_dir"]
    train_sums_dir = config["training_summaries"]
    model_dir = config["model_dir"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    model_input_size = config["model_input_size"]
    alpha = config["alpha"]
    architecture = config["architecture"]
    num_classes = config["num_classes"]
    num_train_examples = config["num_train_examples"]
    num_test_examples = config["num_test_examples"]

    # create training_summaries dir if it doesn't exist
    fu.init_directory(directory=train_sums_dir)

    IMG_SHAPE = (model_input_size, model_input_size, 3)

    # gather training TFRecord files
    filepaths_train = []
    for i in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, i)) and 'train' in i:
            filepaths_train.append(os.path.join(data_dir, i))

    # gather testing TFRecord files
    filepaths_test = []
    for i in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, i)) and 'train' in i:
            filepaths_test.append(os.path.join(data_dir, i))

    # get train and validation data tensors
    image_train, label_train = create_dataset(filepaths=filepaths_train,
                                              batch_size=batch_size,
                                              num_classes=num_classes,
                                              tgt_size=model_input_size)

    image_val, label_val = create_dataset(filepaths=filepaths_test,
                                          batch_size=batch_size,
                                          num_classes=num_classes,
                                          tgt_size=model_input_size)

    # Create the base model from a pre-trained model

    # create input layer
    input_tensor = keras.layers.Input(tensor=image_train)

    # construct base, pretrained model
    base_model = None
    if architecture == MOBILENET_V2:
        base_model = tf.keras.applications.MobileNetV2(input_tensor=input_tensor,
                                                       input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet',
                                                       alpha=alpha)

    # setup tensorboard logging callback
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=train_sums_dir
    )

    # define training callbacks
    callbacks = [
        tensorboard_cb
    ]

    ## Fine-tune last layer

    # freeze model's base layers
    base_model.trainable = False

    # look at the base model architecture
    base_model.summary()

    # add new classification head
    model_k = tf.keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=num_classes, activation='softmax')
    ])

    # compile the model
    model_k.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                    loss='categorical_crossentropy',
                    metrics=[tf.keras.metrics.CategoricalAccuracy()],
                    target_tensors=[label_train])

    # get new model summary
    model_k.summary()

    # train the model
    steps_per_epoch = num_train_examples // batch_size
    validation_steps = num_test_examples // batch_size

    history = model_k.fit(x=image_train,
                          y=label_train,
                          epochs=epochs,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          validation_data=(image_val, label_val),
                          callbacks=callbacks,
                          workers=4)

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
    plt.savefig(os.path.join(train_sums_dir, 'acc_loss_for_{}_layers_fine_tune.png'.format(1)))
    # plt.show()

    # save intermediate model
    model_k.save(filepath=os.path.join(model_dir, 'retrained_model_last_{}_layers.hdf'.format(1)))

    ## Fine-tune more layers

    # allow all layers to be trainable
    base_model.trainable = True

    # let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # fine tune from this layer onwards
    fine_tune_at = 100

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
    history_fine = model_k.fit(x=image_train,
                               y=label_train,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               validation_steps=validation_steps,
                               validation_data=(image_val, label_val),
                               callbacks=callbacks,
                               workers=4)

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
    plt.savefig(os.path.join(train_sums_dir, 'acc_loss_for_last_{}_layers_fine_tune.png'.format(fine_tune_at)))
    # plt.show()

    # save intermediate model
    model_k.save(filepath=os.path.join(model_dir, 'retrained_model_last_{}_layers.hdf'.format(fine_tune_at)))