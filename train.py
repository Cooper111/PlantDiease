import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.utils import multi_gpu_model
from fuck_model import get_best_model,bulid_model
from config import batch_size, patience, num_train_samples, num_valid_samples, num_epochs, verbose
from data_generator import DataGenSequence
from utils import get_available_gpus, ensure_folder, get_highest_acc

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if __name__ == '__main__':
    ensure_folder('models')
    best_model, epoch = get_best_model('/home/yjz/skw/ronghe_2/models')
    if best_model is None:
        initial_epoch = 0
    else:
        initial_epoch = epoch + 1


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = 'models/model.%02d-%.4f.hdf5'
            highest_acc = get_highest_acc()
            if float(logs['val_acc']) > highest_acc:
                self.model_to_save.save(fmt % (epoch, logs['val_acc']))


    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.5, patience=int(patience / 4), verbose=1)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)

    num_gpu = 1  # fix
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            model = bulid_model()


        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        new_model = bulid_model()


    sgd = keras.optimizers.SGD(lr=2.5e-5, momentum=0.9, decay=1e-6, nesterov=True)
    new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # fine tune the model
    new_model.fit_generator(
        DataGenSequence('train'),
        steps_per_epoch=num_train_samples // batch_size,
        validation_data=DataGenSequence('valid'),
        validation_steps=num_valid_samples // batch_size,
        shuffle=True,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=verbose)
