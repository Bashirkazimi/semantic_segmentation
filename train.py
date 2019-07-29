"""
Author: Bashir Kazimi
Data: 26.07.2019
"""

import argparse
from src import dl_model, dataset, utils
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd

# labels for input images
CLASSES = ["bg", "bombs", "meiler", "barrows", "pinge"]

parser = argparse.ArgumentParser(
    description='Arguments for training a semantic segmentation model!')
parser.add_argument('--train_image_dir', type=str, default='/media/kazimi/Data/data/bmbp_data/x',
                    help='Path to input image directory for train set')
parser.add_argument('--train_label_dir', type=str,
                    default='/media/kazimi/Data/data/bmbp_data/y',
                    help='Path to input image labels directory')
parser.add_argument('--validation_image_dir', type=str, default='/media/kazimi/Data/data/bmbp_data/validation/x',
                    help='Path to input image directory for validation set')
parser.add_argument('--validation_label_dir', type=str,
                    default='/media/kazimi/Data/data/bmbp_data/validation/y',
                    help='Path to input image labels directory')
parser.add_argument('--image_dim', type=int, default=128, help='Image dimension')
parser.add_argument('--num_channels', default=1, help='Number of input channels')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for the model')
parser.add_argument('--loss', type=str, default='sparse_categorical_crossentropy', help='objective function')
parser.add_argument('--preprocess', type=bool, default=False,
                    help='True if input image should be preprocessed, do not add flag if no preprocessing.')
parser.add_argument('--metric', type=str, default='sparse_mean_iou', help='Metrics for training')
parser.add_argument('--models_dir', type=str, default='./saved_models', help='Directory to save trained models')
parser.add_argument('--model_name', type=str, default='model', help='Name for saved models/files')
parser.add_argument('--files_dir', type=str, default='./saved_files', help='Directory to save files/images')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size during training')
parser.add_argument('--checkpoint', type=bool, default=True,
                    help='If true, model checkpointed to save the best model weights')
parser.add_argument('--count_mid_flow', type=int, default=8, help='Count of middle layers in deep lab model')
parser.add_argument('--wildcard', default="*.tif", type=str, help='Wildcard to check for files with this extension')


def create_compile_train(args):
    """
    create and compile model
    :return: compiled model
    """
    if args.metric == 'sparse_mean_iou':
        ious = utils.sparse_build_iou_for(list(range(len(CLASSES))), CLASSES)
        ious.append(utils.sparse_mean_iou)
        metrics = ious
    else:
        metrics = None

    model = dl_model.get_model(input_shape=(args.image_dim, args.image_dim, args.num_channels),
                               output_dims=(args.image_dim, args.image_dim), class_no=len(CLASSES),
                               count_mid_flow=args.count_mid_flow)
    model.summary()
    model.compile(loss=args.loss, optimizer=args.optimizer,
                  metrics=metrics)

    train_data = dataset.Data(args.train_image_dir,
                              preprocess=args.preprocess,
                              label_dir=args.train_label_dir,
                              num_classes=len(CLASSES),
                              wildcard=args.wildcard,
                              image_size=args.image_dim,
                              num_channels=args.num_channels,
                              batch_size=args.batch_size
                              )
    valid_data = dataset.Data(args.validation_image_dir,
                              preprocess=args.preprocess,
                              label_dir=args.validation_label_dir,
                              num_classes=len(CLASSES),
                              wildcard=args.wildcard,
                              image_size=args.image_dim,
                              num_channels=args.num_channels,
                              batch_size=args.batch_size
                              )

    if args.checkpoint:
        callbacks = []
        filepaths = ['val_loss_{}'.format(args.model_name), 'loss_{}'.format(args.model_name),
                     'val_metric_{}'.format(args.model_name), 'metric_{}'.format(args.model_name)]
        monitors = ['val_loss', 'loss', 'val_{}'.format(args.metric), args.metric]
        for mon, fp in zip(monitors, filepaths):
            fp = os.path.join(args.models_dir, fp)
            mode = 'max' if 'iou' in mon else 'min'
            cb = ModelCheckpoint(fp, monitor=mon, verbose=0, save_best_only=True,
                                 save_weights_only=True, mode=mode)
            callbacks.append(cb)
    else:
        callbacks = None
    hist = model.fit_generator(generator=train_data, validation_data=valid_data,
                               epochs=args.epochs, verbose=1,
                               callbacks=callbacks)

    df = pd.DataFrame(hist.history)
    df.to_csv(os.path.join(args.files_dir, '{}.csv'.format(args.model_name)))

    model.save_weights(os.path.join(args.models_dir, 'final_{}.h5'.format(args.model_name)))
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(k, v)
    model = create_compile_train(args)
