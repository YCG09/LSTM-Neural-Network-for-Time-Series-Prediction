import os
import json
import math
from core.data_processor import DataLoader
from core.model import Model
from core.utils import Plot


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])
    if not os.path.exists(configs['model']['log_dir']):
        os.makedirs(configs['model']['log_dir'])

    data_loader = DataLoader(
        os.path.join('data', configs['data']['filename_train']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        is_training=True
    )

    model = Model()
    model.build_model(configs)
    steps_per_epoch = math.ceil((data_loader.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    validation_steps = math.ceil((data_loader.len_val - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        train_loader=data_loader.batch_generator(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise'],
            generator_type='train'),
        val_loader=data_loader.batch_generator(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise'],
            generator_type='val'),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        save_dir=configs['model']['save_dir'],
        log_dir=configs['model']['log_dir']
    )

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
