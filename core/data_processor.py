import numpy as np
import pandas as pd


class DataLoader:
    """
    A class for loading and transforming data for the lstm model
    """
    def __init__(self, filename, split, cols, is_training):
        dataframe = pd.read_csv(filename)
        if is_training:
            i_split = int(len(dataframe) * split)
            self.data_train = dataframe.get(cols).values[:i_split]
            self.data_val = dataframe.get(cols).values[i_split:]
            self.len_train = len(self.data_train)
            self.len_val = len(self.data_val)
        else:
            self.data_test = dataframe.get(cols).values[:]
            self.len_test = len(self.data_test)

    def batch_generator(self, seq_len, batch_size, normalise, generator_type):
        """
        Yield a generator of training data from filename on given list of cols split for train/val
        """
        if generator_type == 'train':
            data_len = self.len_train
        elif generator_type == 'val':
            data_len = self.len_val
        else:
            raise ValueError("generator_type must be train or val")

        i = 0
        while i < (data_len - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (data_len - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self.next_window(i, seq_len, normalise, generator_type)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def get_test_data(self, seq_len, normalise):
        """
        Create x, y test data windows
        Warning: not generative, make sure you have enough memory to load data
        """
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)  # shape: [sample_size, sequence_length, embedding_size]
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]  # shape: [sample_size, timestep_size, embedding_size]
        y = data_windows[:, -1, [0]]  # shape: [sample_size, 1]

        return x, y

    def next_window(self, i, seq_len, normalise, generator_type):
        """
        Generates the next data window from the given index location i
        """
        if generator_type == 'train':
            data = self.data_train
        else:
            data = self.data_val

        window = data[i:i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]  # shape: [timestep_size, embedding_size]
        y = window[-1, [0]]  # shape: [1]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        """
        Normalise window with a base value of zero
        """
        normalised_data = []
        window_data = [window_data] if single_window else window_data  # train: 2-d(true), test: 3-d(false)
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):  # multi-dimensional normalise
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            # reshape and transpose array back into original multidimensional format
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
