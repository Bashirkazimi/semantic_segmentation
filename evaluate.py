"""
Author: Bashir Kazimi
Data: 26.07.2019
"""

import argparse
from src import dl_model, dataset, utils
import os
from glob import glob
import numpy as np

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
parser.add_argument('--count_mid_flow', type=int, default=16, help='Count of middle layers in deep lab model')
parser.add_argument('--wildcard', default="*.npy", type=str, help='Wildcard to check for files with this extension')
parser.add_argument('--val', type=bool, default=False,
                    help='evaluating using validation weights')
parser.add_argument('--final', type=bool, default=False,
                    help='evaluating using final weights')
parser.add_argument('--loss_or_not', type=bool, default=False,
                    help='evaluating using loss or metric')
parser.add_argument('--dataproduct', type=str, default='dem', help='Name for for data product')
parser.add_argument('--main_path', type=str, default='/notebooks/tmp/data/bmbp_data', help='main path for the data '
                                                                                           'product data')


def create_compile_evaluate(args):
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

    weight_file = utils.get_weight_file(args.dataproduct, args.val, args.final, args.loss_or_not, args.models_dir)
    model.loss_weights(weight_file)
    model.compile(loss=args.loss, optimizer=args.optimizer,
                  metrics=metrics)

    test_data = dataset.Data(args.validation_image_dir,
                             preprocess=args.preprocess,
                             label_dir=args.validation_label_dir,
                             num_classes=len(CLASSES),
                             wildcard=args.wildcard,
                             image_size=args.image_dim,
                             num_channels=args.num_channels,
                             batch_size=args.batch_size
                             )

    print(model.evaluate_generator(test_data))
    test_data_ids = test_data.ids
    test_prediction = model.predict_generator(test_data)
    test_prediction = np.argmax(test_prediction, -1)
    npy_dir = '{}/test/{}/results/npy_masks'.format(args.main_path, args.dataproduct)
    shp_dir = '{}/test/{}/results/shp_files'.format(args.main_path, args.dataproduct)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    if not os.path.exists(shp_dir):
        os.makedirs(shp_dir)
    i = 0
    for tid in test_data_ids:
        f = test_data.datafiles[tid]
        npy_fn = utils.get_file_name(npy_dir, f, '.npy')
        shp_fn = utils.get_file_name(shp_dir, f, '.shp')
        a = np.zeros((256,256))
        a[64:192,64:192] = test_prediction[i]
        utils.mask_to_polygon(f, a, shp_fn, 'class')
        np.save(npy_fn, test_prediction[i])
        i+=1
    outputMergefn = '{}/test/{}/results/all_masks.shp'.format(args.main_path, args.dataproduct)
    shp_files = glob(os.path.join(shp_dir, '*.shp'))
    utils.merge_shp_files(shp_files, outputMergefn)


if __name__ == '__main__':
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(k, v)
    create_compile_evaluate(args)