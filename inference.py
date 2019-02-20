import os
import json
from core.data_processor import DataLoader
from core.model import Model
from core.utils import Plot


def main():
    configs = json.load(open('config.json', 'r'))

    model = Model()
    model_fname = 'model-20190220164528-e02-0.00015.h5'
    model_path = os.path.join(configs['model']['save_dir'], model_fname)
    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        raise ValueError("can't find model!")

    test_data_loader = DataLoader(
        os.path.join('data', configs['data']['filename_test']),
        0,
        configs['data']['columns'],
        is_training=False
    )

    x_test, y_test = test_data_loader.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequences_full(x_test, configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x_test)

    plot = Plot()
    plot.plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    # plot.plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
