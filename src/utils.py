import gdal
import numpy as np
from gdalconst import GA_ReadOnly
import keras.backend as K
import ogr
import gdalconst


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