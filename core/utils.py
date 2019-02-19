import datetime as dt
import matplotlib.pyplot as plt


class Timer:
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))


class Plot:
    def __init__(self):
        self.facecolor = 'white'
        self.fig = plt.figure(facecolor=self.facecolor)
        self.ax = self.fig.add_subplot(111)

    def plot_results(self, predicted_data, true_data):
        self.ax.plot(true_data, label='True Data')
        self.ax.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.show()

    def plot_results_multiple(self, predicted_data, true_data, prediction_len):
        self.ax.plot(true_data, label='True Data')
        # Pad the list of predictions to level shift it in the graph to it's correct start
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * prediction_len)]
            self.ax.plot(padding + data, label='Prediction')
            plt.legend()
        plt.show()
