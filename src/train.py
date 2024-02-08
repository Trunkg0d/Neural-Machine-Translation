from utils.metrics import masked_acc, masked_loss
from models.translator import Translator
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Setting this env variable prevents TF warnings from showing up

VOCAB_SIZE = 12000
UNITS = 256
def compile_and_train(model, train_data, val_data, epochs=20, steps_per_epoch=500):
    model.compile(optimizer="adam", loss=masked_loss, metrics=[masked_acc, masked_loss])

    history = model.fit(
        train_data.repeat(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )

    return model, history

train_data = tf.data.Dataset.load("../data/processed/train_data")
val_data = tf.data.Dataset.load("../data/processed/val_data")

translator = Translator(VOCAB_SIZE, UNITS)
trained_translator, history = compile_and_train(translator, train_data=train_data, val_data=val_data)

translator.save_weights("../models_weight/test")