import gdal
import numpy as np
from gdalconst import GA_ReadOnly
import keras.backend as K
import ogr
import gdalconst
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os



def read_tif(filename):
    """
    reads a tif file to numpy array and returns
    :param filename: file to read
    :return: numpy array
    """
    raster = gdal.Open(filename, GA_ReadOnly)
    imarray = np.array(raster.ReadAsArray())
    # geotransform = raster.GetGeoTransform()
    return imarray  # , geotransform


def zero_one(im):
    """
    scales the input image between 0 and 1
    :param im: input image
    :return: scaled image
    """
    m = im.min()
    im = (im - m) / (im.max() - m)
    return im


def sparse_iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    y_true = K.squeeze(y_true, axis=-1)
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def sparse_mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + sparse_iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels


def sparse_build_iou_for(label: int, name: str = None):
    """
    Build an Intersection over Union (IoU) metric for a label.
    Args:
        label: the label to build the IoU metric for
        name: an optional name for debugging the built method
    Returns:
        a keras metric to evaluate IoU for the given label

    Note:
        label and name support list inputs for multiple labels
    """
    # handle recursive inputs (e.g. a list of labels and names)
    if isinstance(label, list):
        if isinstance(name, list):
            return [sparse_build_iou_for(l, n) for (l, n) in zip(label, name)]
        return [sparse_build_iou_for(l) for l in label]

    # build the method for returning the IoU of the given label
    def label_iou(y_true, y_pred):
        """
        Return the Intersection over Union (IoU) score for {0}.
        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or softmax output
        Returns:
            the scalar IoU value for the given label ({0})
        """.format(label)
        return sparse_iou(y_true, y_pred, label)

    # if no name is provided, us the label
    if name is None:
        name = label
    # change the name of the method for debugging
    label_iou.__name__ = 'sparse_iou_{}'.format(name)

    return label_iou


def sliding_window(image, step_size, windowSize):
    """
    Scan image with a stride of step_size and patches of (windowSize X windowSize)
    return a generator
    """
    for i in range(0, image.shape[0], step_size):
        for j in range(0, image.shape[1], step_size):
            im = image[i:i + windowSize, j:j + windowSize]
            if im.shape == (windowSize,windowSize):
                yield (i, j, im)


def mask_to_polygon(tif_file, array, output_shp_file, field_name=None):
    """
    Takes a numpy array, creates a shape file polygon with projections and coordinates info from tif_file
    saving values to field_name attribute.
    :param tif_file: tif_file with projection info
    :param array: the array to turn into polygons
    :param output_shp_file: path to save the shapefile
    :param field_name: field name to save values from array
    :return:
    """
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create('', array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(array)
    original_data = gdal.Open(tif_file, gdalconst.GA_ReadOnly)
    geotrans = original_data.GetGeoTransform()
    proj = original_data.GetProjection()
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    original_data = None
    band = dataset.GetRasterBand(1)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    outDatasource = driver.CreateDataSource(output_shp_file)
    outLayer = outDatasource.CreateLayer("polygonized", srs=None)
    if field_name is None:
        field_name='MyFLD'
    newField = ogr.FieldDefn(field_name, ogr.OFTInteger)
    outLayer.CreateField(newField)

    driver_mask = gdal.GetDriverByName('MEM')
    ds_mask = driver_mask.Create('', array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    ds_mask.SetGeoTransform(geotrans)
    ds_mask.SetProjection(proj)
    ds_mask_array = (array>0).astype(np.int32)
    ds_mask.GetRasterBand(1).WriteArray( ds_mask_array )
    mask_band = ds_mask.GetRasterBand(1)

    gdal.Polygonize(band, mask_band, outLayer, 0, [], callback=None )
    outDatasource.Destroy()
    sourceRaster = None
    dataset.FlushCache()
    dataset = None


def run_detection_save_to_shps2(batch_size,
                                tif_file,
                                windowSize,
                                output_patch,
                                step_size,
                                model,
                                original_tif,
                                path_shp_file,
                                conf_threshold=0.90):
    """
    Takes a big tif_file, scans it with a window of windowSize x windowSize with a stride of step_size,
    makes predictions with the model using batch_size examples each. Finally it creates a polygon from the results
    and saves the result to path_shp_file. While predictions, it only counts predictions with probability above
    conf_threshold, otherwise it labels it as background.
    :param batch_size:
    :param tif_file:
    :param windowSize:
    :param output_patch:
    :param step_size:
    :param model:
    :param original_tif:
    :param path_shp_file:
    :param conf_threshold:
    :return:
    """
    counter = 0 # keep track of number of examples
    batch_input_index = 0 # keep track of examples added to current batch
    top_indices = [] # keep track of top left indices for each input

    # read test region to numpy array
    tif_data = read_tif(tif_file)
    if tif_data.shape != (2016, 2016):
        return
    cropped_data = tif_data[48:1920+48,48:1920+48]
    gen = sliding_window(cropped_data, step_size, windowSize)
    final_array = np.zeros(tif_data.shape)

    # Create an empty numpy array of shape similar to input test region
    test_result = np.zeros(cropped_data.shape)

    for i, j, im in gen:
        top_indices.append((i, j))
        if counter % batch_size == 0:
            batch_input = np.empty((batch_size, windowSize, windowSize))
        batch_input[batch_input_index] = im
        batch_input_index += 1
        counter += 1
        if batch_input_index == batch_size:
            predictions = model.predict(np.expand_dims(batch_input, -1))
            maxed_p = np.max(predictions, -1)
            masked_p = maxed_p >= conf_threshold
            argmaxed_p = np.argmax(predictions, axis=-1)
            predictions = argmaxed_p*masked_p.astype(np.int)
            p = 0
            for r, c in top_indices:
                print(r, c)
                test_result[r:r+output_patch[0], c:c+output_patch[1]] = predictions[p]
                p += 1
            top_indices = []
            batch_input_index = 0

    if batch_input_index > 0:
        predictions = model.predict(np.expand_dims(batch_input, -1))
        maxed_p = np.max(predictions, -1)
        masked_p = maxed_p >= conf_threshold
        argmaxed_p = np.argmax(predictions, axis=-1)
        predictions = argmaxed_p*masked_p.astype(np.int)
        p = 0
        for r, c in top_indices:
            test_result[r:r+output_patch[0], c:c+output_patch[1]] = predictions[p]
            p += 1
    final_array[48:1920+48,48:1920+48] = test_result
    mask_to_polygon(original_tif, final_array, path_shp_file, field_name='label')


def write_numpy_array_to_tif(numpyarray, tifname, original_tif=None):
    """
    Takes a numpy array and saves it in tif format at tifname location. The projections are given by originital_tif
    if given
    :param numpyarray: numpy array to save to tif file
    :param tifname: where to save the tif file
    :param original_tif: projections to use
    :return:
    """
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(tifname, numpyarray.shape[1], numpyarray.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(numpyarray)
    if original_tif is not None:
        original_data = gdal.Open(original_tif, gdalconst.GA_ReadOnly)
        geotrans = original_data.GetGeoTransform()
        proj = original_data.GetProjection()
        dataset.SetGeoTransform(geotrans)
        dataset.SetProjection(proj)
        original_data = None
    dataset.FlushCache()
    dataset = None


def plot_loss_val_loss(filename):
    """
    Takes a csv file with model history and plots loss vs val_loss. Saves the plot to the same path as filename in
    png format.
    :param filename: path to the csv file with model history
    :return:
    """
    df = pd.read_csv(filename)
    pth, fn = os.path.split(filename)
    fn, ext = os.path.splitext(fn)
    plt.rcParams.update({'font.size': 20})
    df[['loss', 'val_loss']].plot(figsize=(15, 10))
    plt.title('Train and validation loss for {} data'.format(fn))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(pth, fn+'.png'))


def all_loss(filenames, val=False, offset=100, metric_name='loss'):
    """
    Takes a list of csv files with histories for the same kind of metrics/loss. It plots the metric/loss for training
    or validation until offset epochs
    :param filenames: list of csv files with histories
    :param val: if True, plots validation metric/loss
    :param offset: until which epoch to plot
    :param metric_name: values to plot
    :return:
    """
    names_list = []
    value_list = []
    pth_to_save = None
    for fn in filenames:
        df = pd.read_csv(fn)
        if val:
            value_list.append(list(df['val_{}'.format(metric_name)]))
        else:
            value_list.append(list(df[metric_name]))
        pth, fn = os.path.split(fn)
        pth_to_save = pth
        fn, ext = os.path.splitext(fn)
        names_list.append(fn)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(15, 10))
    for i in range(len(names_list)):
        plt.plot(list(range(len(value_list[i][:offset]))), value_list[i][:offset], label=names_list[i], linewidth=3)
    title = '{} {} comparison'.format('Validation' if val else 'Train', metric_name)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    fn = 'all_val' if val else 'all_train'
    plt.savefig(os.path.join(pth_to_save, fn+metric_name+str(offset)+'.png'))


def merge_shp_files(shp_files, outputMergfn):
    """
    Takes a list of shp_files and merges them all and saves it at ouputMergfn shape file
    :param shp_files: list of shapefiles to merge
    :param outputMergfn: where to save the merged shapes
    :return:
    """
    driverName = 'ESRI Shapefile'
    geometryType = ogr.wkbPolygon
    out_driver = ogr.GetDriverByName( driverName )
    if os.path.exists(outputMergfn):
        out_driver.DeleteDataSource(outputMergfn)
    out_ds = out_driver.CreateDataSource(outputMergfn)
    out_layer = out_ds.CreateLayer(outputMergfn, geom_type=geometryType)
    only_ones = True
    for file in shp_files:
        ds = ogr.Open(file)
        lyr = ds.GetLayer()
        if only_ones:
            lyr_def = lyr.GetLayerDefn ()
            for i in range(lyr_def.GetFieldCount()):
                out_layer.CreateField (lyr_def.GetFieldDefn(i) )
            only_ones=False
        for feat in lyr:
            out_layer.CreateFeature(feat)


def double_sparse_iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(y_pred, label), K.floatx())
    # calculate the |intersection| (AND) of the labels
#     y_true = K.squeeze(y_true,axis=-1)
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def double_sparse_mean_iou(num_labels=5):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    # initialize a variable to store total IoU in
    def dsmiou(y_true, y_pred):
        total_iou = K.variable(0)
        # iterate over labels to calculate IoU for
        for label in range(num_labels):
            total_iou = total_iou + double_sparse_iou(y_true, y_pred, label)
        # divide total IoU by number of labels to get mean IoU
        return total_iou / num_labels
    return dsmiou


def get_weight_file(d, val=False, final=False, loss=True, model_path='./saved_models'):
    """
    Gets the best weight file for a model with a certain data product
    :param d: one of ['dem', 'slrm', etc]
    :param val: if true, get best validation weights, else train weights
    :param final: if true, get the final weight file
    :param loss: if true, get the loss weights, else metric weights
    :param model_path: folder with weight files
    :return:
    """
    if final:
        fn = "final_{}.h5".format(d)
        fn = os.path.join(model_path, fn)
        return fn
    if val:
        if loss:
            fn = "val_loss_{}".format(d)
        else:
            fn = "val_metric_{}".format(d)
        fn = os.path.join(model_path, fn)
        return fn
    else:
        if loss:
            fn = "loss_{}".format(d)
        else:
            fn = "metric_{}".format(d)
        fn = os.path.join(model_path, fn)
        return fn


def get_file_name(output_path, f, my_ext):
    """
    Creates a file name for npy or shp file using the filename taking its specific number
    :param output_path: where to save the file
    :param f: original file name
    :param my_ext: npy or shp
    :return: new file name
    """
    pth, name = os.path.split(f)
    just_name, ext = os.path.splitext(name)
    niedersachsen_num = just_name.split('_')[0]
    just_num = niedersachsen_num[13:]
    new_name = just_num+my_ext
    return os.path.join(output_path, new_name)
