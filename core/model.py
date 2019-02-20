import os
import time
import numpy as np
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard


class Model:
    """
    A class for an building and inferencing an lstm model
    """
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        print('[Model] Model Compiled')
        timer.stop()

    def train_generator(self, train_loader, val_loader, epochs, batch_size, steps_per_epoch, validation_steps, save_dir, log_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        time_format = time.localtime(time.time())
        model_fname = 'model-%s-e{epoch:02d}-{val_loss:.5f}.h5' % (time.strftime('%Y%m%d%H%M%S', time_format))
        model_save_path = os.path.join(save_dir, model_fname)
        lr_schedule = lambda epoch: 0.001 * 0.95 ** epoch
        learning_rate = np.array([lr_schedule(i) for i in range(epochs)])
        callbacks = [
            ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=False),
            LearningRateScheduler(lambda epoch: float(learning_rate[epoch])),
            EarlyStopping(monitor='val_loss', patience=2, verbose=1),
            TensorBoard(log_dir=log_dir, write_graph=True)
        ]
        self.model.fit_generator(train_loader,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 validation_data=val_loader,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks)
        print('[Model] Training Completed. Model saved as %s' % model_save_path)
        timer.stop()

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)  # shape: [n, 1]
        predicted = np.reshape(predicted, (predicted.size,))  # shape: [n]
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]  # shape: [timestep_size, embedding_size]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                # Remove the head of time sequence, then append the latest prediction result at the end of sequence
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequences_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            # Remove the head of time sequence, then append the latest prediction result at the end of sequence
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
