from __future__ import absolute_import, division, print_function

import tensorflow as tf
from ResNet.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
import config
from prepare_data import generate_datasets_60x, generate_datasets_20x
import math
import csv


def get_model():
    model = resnet_50()
    if config.model == "resnet18":
        model = resnet_18()
    if config.model == "resnet34":
        model = resnet_34()
    if config.model == "resnet101":
        model = resnet_101()
    if config.model == "resnet152":
        model = resnet_152()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    return model


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    if config.TIMES == 20:
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets_20x()
    elif config.TIMES == 60:
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets_60x()
    else:
        raise ValueError(f"Image magnification in config.py should be 20 or 60, got {config.TIMES} instead")

    # create model
    model = get_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    global_steps = tf.Variable(0, trainable=False)

    # create learning rate optimizer
    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=config.LEARNING_RATE, decay_steps=1000, decay_rate=0.01)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


    @tf.function(experimental_relax_shapes=True)
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)
        valid_loss(v_loss)
        valid_accuracy(labels, predictions)


    train_acces = []
    train_losses = []
    val_acces = []
    val_losses = []
    train_steps = []
    valid_steps = []
    best_acc = None

    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            train_step(images, labels)
            step += 1
            print("\repoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}"
                  .format(epoch + 1,
                          config.EPOCHS,
                          step,
                          math.ceil(train_count / config.BATCH_SIZE),
                          train_loss.result(),
                          train_accuracy.result()), end='')
            train_steps.append(step * (1 + epoch))
            train_acces.append(float(train_accuracy.result()))
            train_losses.append(float(train_loss.result()))

        val_step = 0
        valid_acc = None
        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)
            valid_steps.append(val_step)
            val_acces.append(float(valid_accuracy.result()))
            val_losses.append(float(valid_loss.result()))
            valid_acc =  valid_accuracy.result()
            print("\rEpoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                  "valid loss: {:.5f}, valid accuracy: {:.5f}"
                  .format(epoch + 1,
                          config.EPOCHS,
                          train_loss.result(),
                          train_accuracy.result(),
                          valid_loss.result(),
                          valid_accuracy.result()), end='')
            val_step += 1
        if valid_acc > 0.8:
            if best_acc is None:
                print(f'\nnow best accuracy is {valid_acc}')
                best_acc = valid_acc
                model.save_weights(filepath=config.save_model_dir_20x_best, save_format='tf')
            else:
                if best_acc < valid_acc:
                    best_acc = valid_acc
                    print(f'now best accuracy is {valid_acc}')
                    model.save_weights(filepath=config.save_model_dir_20x_best, save_format='tf')
        print()

    with open(config.train_process_20x_detail_data_savefile, 'w',
              newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(train_steps)
        writer.writerow(train_acces)
        writer.writerow(train_losses)
        writer.writerow(valid_steps)
        writer.writerow(val_acces)
        writer.writerow(val_losses)

    model.save_weights(filepath=config.save_model_dir_20x, save_format='tf')
