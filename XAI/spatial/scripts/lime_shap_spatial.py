"""
### ***Cineca***
"""

import tensorflow as tf
#from tensorflow.keras import layers, activations, callbacks, models
import numpy as np
import pickle
import os
from keras.models import load_model
from skimage.transform import resize
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import pickle
import numpy as np
import geopandas as gpd
import xarray
import rioxarray
from skimage.segmentation import slic
import sys
# Save Execution Time
import datetime

"""
##### ***Data & Black-Box***

"""

RESULT_DIR = str(sys.argv[1])
print(f"RESULT_DIR: {RESULT_DIR}")

# IMPORTO I DATI PER VOTTIGNASCO
# Ottieni il percorso effettivo da una variabile d'ambiente
work_path = os.environ['WORK']  # Ottieni il valore della variabile d'ambiente WORK
v_test_OHE_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_month_OHE.npy")
v_test_image_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_normalized_image_sequences.npy")
v_test_target_dates_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_target_dates.npy")
shapefile_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/shapefile_raster/")
v_test_normalization_factors_std_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_training_target_std.npy")
v_test_normalization_factors_mean_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_training_target_mean.npy")

# Carica l'array numpy dai file
vottignasco_test_OHE    = np.load(v_test_OHE_path)
vottignasco_test_image  = np.load(v_test_image_path)
vottignasco_test_dates  = np.load(v_test_target_dates_path)
vott_target_test_std    = np.load(v_test_normalization_factors_std_path) 
vott_target_test_mean   = np.load(v_test_normalization_factors_mean_path)


print(len(vottignasco_test_dates))
print(len(vottignasco_test_image))
print(len(vottignasco_test_OHE))

# """##### ***Black Boxes***"""

# Enable dropout at runtime if needed
mc_dropout = True

# Definition of the custom dropout class
class doprout_custom(tf.keras.layers.SpatialDropout1D):
    def call(self, inputs, training=None):
        if mc_dropout:
            return super().call(inputs, training=True)
        else:
            return super().call(inputs, training=False)

# Path to the directory on Cineca
base_dir = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco")
lstm_suffix = 'time_dist_LSTM'

vott_lstm_models = []

def extract_index(filename):
    """Function to extract the final index from the filename."""
    return int(filename.split('_LSTM_')[-1].split('.')[0])

# Find all .keras files in the folder and add them to the list
for filename in os.listdir(base_dir):
    if lstm_suffix in filename and filename.endswith(".keras"):
        vott_lstm_models.append(os.path.join(base_dir, filename))

# Sort models based on the final index
vott_lstm_models = sorted(vott_lstm_models, key=lambda x: extract_index(os.path.basename(x)))

# List to store the loaded models
vott_lstm_models_loaded = []

for i, model_lstm_path in enumerate(vott_lstm_models[:10]):  # Load the first 10 sorted models
    #print(f"Loading LSTM model {i+1}: {model_lstm_path}")

    # Load the model with the custom class
    model = load_model(model_lstm_path, custom_objects={"doprout_custom": doprout_custom})

    # Add the model to the list
    vott_lstm_models_loaded.append(model)

print(vott_lstm_models_loaded)

"""### ***Spatial-LIME***

#### ***Spatial-Superpixels***
"""

def create_spatial_superpixels(shapefile_path, n_segments=8, compactness=15):
    # DTM [50m] import
    dtm_piemonte = rioxarray.open_rasterio(shapefile_path + 'DTMPiemonte_filled_50m.tif')
    dtm_piemonte = dtm_piemonte.rio.reproject("epsg:4326")
    dtm_piemonte = dtm_piemonte.where(dtm_piemonte != -99999)  # Keep only valid pixels

    # Catchment shapefile
    catchment = gpd.read_file(shapefile_path + "BAC_01_bacialti.shp")  # Select GRANA-MAIRA and VARAITA
    catchment = catchment.to_crs('epsg:4326')

    # Select only the Grana-Maira catchment
    catchment_GM = catchment.loc[catchment.NOME == "GRANA-MAIRA"]
    catchment_GM = catchment_GM.reset_index(drop=True)

    # Retrieve the catchment boundaries from the shapefile
    xmin_clip, ymin_clip, xmax_clip, ymax_clip = catchment_GM.total_bounds

    # Extend the boundaries to include more pixels at the edges
    increase = 0.05  # Degrees
    #ymin_clip -= increase  # Not needed
    xmin_clip += increase  # "+" to subset pixels included in the mask
    xmax_clip += increase
    #ymax_clip += increase  # Not needed

    dtm_piemonte_clipped = dtm_piemonte.rio.clip_box(minx=xmin_clip, maxx=xmax_clip, miny=ymin_clip, maxy=ymax_clip)

    # Create a 5x8 image with lat, lon, and dtm values
    # Define coordinates
    lon = np.array([6.938, 7.063, 7.188, 7.313, 7.438, 7.563, 7.688, 7.813])  # 8 values
    lat = np.array([44.313, 44.438, 44.563, 44.688, 44.813])  # 5 values

    # Create a 5x8 lat-lon grid
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Create a 5x8x3 array
    img = np.zeros((5, 8, 3))

    # Assign coordinates to the first two channels
    img[:, :, 0] = lat_grid  # Channel 0 = latitude
    img[:, :, 1] = lon_grid  # Channel 1 = longitude
    img[:, :, 2] = 0  # Channel 2 = placeholder value

    for nr_lat, latitude in enumerate(lat):
        for nr_lon, longitude in enumerate(lon):
            img[nr_lat, nr_lon, 2] = dtm_piemonte_clipped.sel(x=longitude, y=latitude, method='nearest').values

    img = np.nan_to_num(img, nan=0.0)

    # SLIC segmentation
    segments = slic(img, n_segments=n_segments, compactness=compactness)

    # Create Spatial-Superpixels
    # Find unique values in the matrix (the clusters)
    clusters = np.unique(segments)

    # Create a list of binary matrices for each cluster
    binary_matrices = {}

    for cluster in clusters:
        binary_matrices[cluster] = (segments == cluster).astype(int)

    spatial_superpixels = [matrix for _, matrix in binary_matrices.items()]

    spatial_superpixel_clusters = []

    for ss in spatial_superpixels:
        indices = np.argwhere(ss == 1)
        cluster_pixels = [(x, y) for x, y in indices]
        spatial_superpixel_clusters.append(cluster_pixels)

    return spatial_superpixels, spatial_superpixel_clusters, segments

"""#### ***Interpretable Representations***"""

# Creation of z'

import itertools
import numpy as np

def generate_z_primes(n):
    """Args:
        n (int): Number of considered superpixels
    Returns:
        np.array: All possible combinations of 0s and 1s of length n
    """

    # Generate all possible combinations of 0s and 1s of length n
    z_primes = list(itertools.product([0, 1], repeat=n))
    # Convert tuples into a numpy array (optional)
    z_primes = np.array(z_primes)
    # Remove the first element (all zeros)
    z_primes = z_primes[1:]
    # Remove the last element (all ones)
    z_primes = z_primes[:-1]
    
    return z_primes

"""#### ***Generation & Application of Uniform Noise Masks (2D)***"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_masks(superpixels, input_size):
    """Generates masks for each superpixel.

    Args:
        superpixels: List of superpixel segmentations.
        input_size: Tuple (height, width) of the input image.

    Returns:
        masks: A NumPy array of shape (N, height, width) with generated masks.
    """
    N = len(superpixels)
    height, width = input_size
    masks = np.empty((N, height, width))

    for i in tqdm(range(N), desc='Generating masks'):
        mask = np.ones((height, width))
        indices = np.argwhere(superpixels[i] == 1)

        for (y, x) in indices:
            mask[y, x] = 0  # Set cluster pixels to 0

        masks[i] = mask

    return masks

"""#### ***Application of Masks***"""

def multiplicative_uniform_noise_onechannel(instance, zs_primes, masks, channel, std_zero_value=-0.6486319166678826):
    """Applies uniform noise masks to a specific channel of the input instance.

    Args:
        instance: The original input instance.
        zs_primes: Binary vectors indicating which superpixels to mask.
        masks: Generated masks for each superpixel.
        channel: The specific channel to apply the noise to.
        std_zero_value: Default value used for masked areas.

    Returns:
        A list of perturbed instances.
    """
    masked = []
    for z in zs_primes:
        masked_instance = copy.deepcopy(instance)
        for i, z_i in enumerate(z):
            if z_i == 0:
                # Apply perturbation only to the specified channel
                masked_instance[..., channel] = (
                    masked_instance[..., channel] * masks[i] + (1 - masks[i]) * std_zero_value
                )

        masked.append(masked_instance)

    return masked

def ensemble_predict(models, images, x3_exp, batch_size=1000):
    """Performs ensemble prediction using multiple models.

    Args:
        models: List of trained models.
        images: List of input images.
        x3_exp: Additional input tensor.
        batch_size: Number of images to process per batch.

    Returns:
        A NumPy array of the ensemble-averaged predictions.
    """
    # Ensure images is a list
    if not isinstance(images, list):
        images = [images]

    len_x3 = len(images)

    # Convert x3_exp into a tensor replicated for each image
    x3_exp_tensor = tf.convert_to_tensor(x3_exp, dtype=tf.float32)

    # List to store final predictions
    final_preds = []

    # Batch processing
    for i in range(0, len_x3, batch_size):
        batch_images = images[i:i + batch_size]
        batch_len = len(batch_images)

        # Convert batch to tensors
        Y_test = tf.stack([tf.convert_to_tensor(img, dtype=tf.float32) for img in batch_images])
        Y_test_x3 = tf.tile(tf.expand_dims(x3_exp_tensor, axis=0), [batch_len, 1, 1])

        # Collect predictions from all models for the current batch
        batch_preds = []

        for model in models:
            preds = model.predict([Y_test, Y_test_x3], verbose=0)
            batch_preds.append(preds)

        # Convert batch predictions to a tensor and compute the mean
        batch_preds_tensor = tf.stack(batch_preds)
        mean_batch_preds = tf.reduce_mean(batch_preds_tensor, axis=0)

        # Add batch predictions to the final list
        final_preds.extend(mean_batch_preds.numpy())

    return np.array(final_preds)

"""#### ***Calculation of Regressor Weights***

###### ***LIME***
Where *calculate_D*:
* D is the L2-Distance (Euclidean Distance)
* x is the original instance to explain
* z is the perturbed non-interpretable version
"""

def calculate_D(instance, perturbed_instance):
    """Computes the Euclidean distance between the original and perturbed instances.

    Args:
        instance: The original instance.
        perturbed_instance: The perturbed instance.

    Returns:
        The Euclidean distance between the two instances.
    """
    x = instance.flatten()
    z = perturbed_instance.flatten()

    return np.linalg.norm(x - z)

def calculate_weights_lime(instance, perturbed_instances, percentile_kernel_width):
    """Computes weights for LIME using an exponential kernel function.

    Args:
        instance: The original instance.
        perturbed_instances: A list of perturbed instances.
        percentile_kernel_width: The percentile used to determine the kernel width.

    Returns:
        A NumPy array of weights for each perturbed instance.
    """
    distances = [calculate_D(instance, perturbed_instance) for perturbed_instance in perturbed_instances]
    kernel_width = np.percentile(distances, percentile_kernel_width)
    
    # Importance of neighbors
    weights = np.exp(- (np.array(distances) ** 2) / (kernel_width ** 2))
    
    return weights


"""##### ***Kernel-SHAP***"""

import math
from scipy.special import binom

def shap_kernel_weight(M, z):
    """
    Computes the Kernel SHAP weight for a given mask (interpretable instance).

    Args:
        M (int): Total number of features.
        z (array): Array containing a zs_prime.

    Returns:
        float: Kernel weighting value for z'.
    """

    z_size = np.sum(z)
    
    # Edge cases where the mask is empty or full
    if z_size == 0 or z_size == M:
        return 0  # Zero weight in these extreme cases
    
    # Binomial coefficient: M choose subset size (|z'|)
    # Kernel SHAP weight formula
    weight = (M - 1) / (binom(M, z_size) * (z_size * (M - z_size)))
    
    return weight

def calculate_weights_shap(M, zs_primes):
    """
    Computes the Kernel SHAP weights for all perturbed instances.

    Args:
        M (int): Total number of features.
        zs_primes (list): List of perturbed feature subsets.

    Returns:
        np.array: Array of Kernel SHAP weights.
    """
    weights = [shap_kernel_weight(M, z) for z in zs_primes]
    return np.array(weights)

"""#### ***Lime-Spatial: Framework***"""

from sklearn.linear_model import Ridge

def lime_shap(nr_instance, dataset_test_image, dataset_test_OHE, channels, models,
              n_segments, compactness, input_size, H_station=390.0, std_zero_value=-0.6486319166678826):
    """
    Framework for Lime-Spatial to explain a given instance by generating spatial superpixels and applying perturbations.

    Args:
        nr_instance (int): Index of the instance to explain.
        dataset_test_image (list): List of test images.
        dataset_test_OHE (list): List of one-hot encoded month frames for the test images.
        channels (tuple): Tuple containing the channels for precipitation, Tmax, and Tmin.
        models (list): List of trained models for ensemble prediction.
        n_segments (int): Number of superpixels.
        compactness (int): Compactness factor for superpixel creation.
        input_size (tuple): The size of the input image (height, width).
        H_station (float): The station value to normalize the predictions (default 390.0).
        std_zero_value (float): The standard deviation value for zero perturbation (default -0.6486319166678826).

    Returns:
        tuple: A tuple containing superpixels, zs_primes, perturbed_instances, and denormalized predictions.
    """

    channel_prec, _, _ = channels

    # Original instance to explain
    instance = copy.deepcopy(dataset_test_image[nr_instance])
    # One-hot encoded months of the frame for the instance
    x3_instance = copy.deepcopy(dataset_test_OHE[nr_instance])

    # Create Spatial Superpixels
    superpixels, _, _ = create_spatial_superpixels(shapefile_path, n_segments=n_segments, compactness=compactness)
    # Generate interpretable representations of the instance
    zs_primes = generate_z_primes(len(superpixels))
    # Create masks for the superpixels
    masks = generate_masks(superpixels, input_size)
    # Create perturbed instances by applying multiplicative noise
    perturbed_instances = multiplicative_uniform_noise_onechannel(instance, zs_primes, masks, channel_prec, std_zero_value)
    # Perform ensemble prediction on the perturbed instances
    preds_masked = ensemble_predict(models, list(perturbed_instances), x3_instance)
    # Denormalize the predictions based on the output of the black-box model
    denorm_preds_masked = [pred_masked * vott_target_test_std + vott_target_test_mean for pred_masked in preds_masked]
    # Further denormalize predictions with respect to the station value
    denormalized_H_preds_masked = [H_station - denorm_pred_masked for denorm_pred_masked in denorm_preds_masked]

    return superpixels, zs_primes, perturbed_instances, denormalized_H_preds_masked

"""#### ***Evaluation Metrics***"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def calculate_auc(x, y):
    """
    Calculate the area under the curve (AUC) using the trapezoidal method.

    :param x: x-axis values (fraction of pixels/frames inserted).
    :param y: y-axis values (calculated errors).
    :return: Area under the curve.
    """
    return np.trapz(y, x)

import numpy as np

def sorted_per_importance_superpixels_index(array):
    """
    Sort the superpixels by their importance, based on the array values.

    :param array: 2D array containing the superpixels' importance scores.
    :return: List of indices sorted by superpixel importance.
    """
    array = np.array(array)  # Convert to numpy array if it's not already
    unique_values = np.unique(array)  # Find unique values

    # Create a dictionary with values as keys and lists of indices as values
    indices_per_value = {val: [] for val in unique_values}

    # Populate the dictionary with indices
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            indices_per_value[array[i, j]].append((i, j))

    # Sort the values in descending order
    sorted_values = sorted(indices_per_value.keys(), reverse=True)

    # Create the final list of indices grouped by value
    results = [indices_per_value[val] for val in sorted_values]

    return results

"""##### ***Insertion***"""

def update_instance_with_superpixels(current_instance, original_instance, index_of_superpixels):
    """
    Update the image by inserting the most important pixels.

    :param current_instance: Current instance.
    :param original_instance: Original instance.
    :param index_of_superpixels: List containing indices of the superpixels to insert.
    :return: Updated instance with inserted superpixels.
    """
    new_current_instance = current_instance.copy()

    for x, y in index_of_superpixels:
        new_current_instance[:, x, y, 0] = original_instance[:, x, y, 0]
    return new_current_instance

def insertion(models, original_instance, x3, sorted_per_importance_all_superpixels_index, initial_blurred_instance, original_prediction, H_station=390.0):
    """
    Calculate the insertion metric for a given explanation.

    :param models: List of pre-trained models.
    :param original_instance: Original instance.
    :param x3: One-hot encoding for the prediction.
    :param sorted_per_importance_all_superpixels_index: List of all superpixels sorted by importance.
    :param initial_blurred_instance: Initial blurred image (all pixels zero).
    :param original_prediction: Original prediction.
    :param H_station: Station value to normalize predictions (default 390.0).
    :return: List of errors at each insertion step.
    """

    # List to store instances as pixels are gradually added, initialized with the blurred instance
    insertion_images = [initial_blurred_instance]

    # Prediction on the initial image (all pixels zero)
    I_prime = copy.deepcopy(initial_blurred_instance)

    # Gradually add the most important pixels for each frame. Get a list of images with pixels added progressively
    for index_of_superpixels in sorted_per_importance_all_superpixels_index:
        I_prime = update_instance_with_superpixels(I_prime, original_instance, index_of_superpixels)
        insertion_images.append(I_prime)

    # Calculate predictions on the instances where pixels have been gradually added
    new_predictions = ensemble_predict(models, insertion_images, x3)
    denorm_new_predictions = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    
    # Calculate MSE for each prediction compared to the original prediction (as in the test set)
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions[1:]]

    initial_error = mean_squared_error(original_prediction, denormalized_H_new_predictions[0])
    print(f"Initial Prediction with Blurred Instance, new prediction: {denormalized_H_new_predictions[0]}, error: {initial_error}")
    only_inserted_pixel_new_predictions = denormalized_H_new_predictions[1:]

    for nr_superpixel, error in enumerate(errors):
        print(f"SuperPixel: {sorted_per_importance_all_superpixels_index[nr_superpixel]}, new prediction: {only_inserted_pixel_new_predictions[nr_superpixel]}, error: {error}")

    total_errors = [initial_error] + errors  # Initial error + errors for all inserted pixels

    # New X-axis: number of superpixels inserted (1, 2, ..., 8)
    x = np.arange(0, len(total_errors))  # From 0 to 8 inclusive

    x_for_auc = np.linspace(0, 1, len(total_errors))
    # Calculate the AUC with the new x-axis
    auc = calculate_auc(x_for_auc, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    return total_errors, auc

"""##### ***Deletion***"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def update_instance_removing_superpixels(current_instance, index_of_superpixels, std_zero_value=-0.6486319166678826):
    """
    Update the image by removing the most important pixels.

    :param current_instance: Current instance.
    :param original_instance: Original instance.
    :param index_of_superpixels: List containing indices of the superpixels to remove.
    :return: Updated instance with removed superpixels.
    """
    new_current_instance = current_instance.copy()

    for x, y in index_of_superpixels:
        new_current_instance[:, x, y, 0] = std_zero_value
    return new_current_instance

def deletion(models, original_instance, x3_instance, sorted_per_importance_all_superpixels_index, original_prediction, H_station=390.0):
    """
    Calculate the deletion metric for a given explanation.

    :param models: List of pre-trained models.
    :param original_instance: Original instance.
    :param x3_instance: One-hot encoding for the prediction.
    :param sorted_per_importance_all_superpixels_index: List of all superpixels sorted by importance.
    :param original_prediction: Original prediction.
    :param H_station: Station value to normalize predictions (default 390.0).
    :return: List of errors at each deletion step.
    """

    # List to store instances as pixels are gradually removed, initialized with the original instance
    deletion_images = []

    # Prediction on the initial image (all pixels zero)
    I_prime = copy.deepcopy(original_instance)

    # Gradually remove the most important pixels for each frame. Get a list of images with pixels removed progressively
    for index_of_superpixels in sorted_per_importance_all_superpixels_index:
        I_prime = update_instance_removing_superpixels(I_prime, index_of_superpixels)
        deletion_images.append(I_prime)

    # Calculate predictions on the instances where pixels have been gradually removed
    new_predictions = ensemble_predict(models, deletion_images, x3_instance)
    denorm_new_predictions = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    
    # Calculate MSE for each prediction compared to the original prediction
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions]

    initial_error = 0.0
    print(f"Initial Prediction with Original instance, prediction: {original_prediction}, error: {initial_error}")

    for nr_superpixel, error in enumerate(errors):
        print(f"Removed SuperPixel: {sorted_per_importance_all_superpixels_index[nr_superpixel]}, new prediction: {denormalized_H_new_predictions[nr_superpixel]}, error: {error}")

    total_errors = [initial_error] + errors  # Initial error + errors for all removed pixels

    x_for_auc = np.linspace(0, 1, len(total_errors))
    # Calculate the AUC with the new x-axis
    auc = calculate_auc(x_for_auc, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    return total_errors, auc


def calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients, spatial_superpixels, models=vott_lstm_models_loaded, 
                                                           H_station=390.0, channel_prec=0, std_zero_value=-0.6486319166678826,input_size=(104,5,8,3),T=104,H=5,W=8):
  
  instance    = copy.deepcopy(vottignasco_test_image[nr_instance])
  x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])

  abs_coefficients = np.abs(coefficients)

  saliency_map_i     = np.zeros((H,W))
  saliency_map_i_abs = np.zeros((H,W))
  # Create the saliency map with the coefficients in the superpixels
  for i,superpixel in enumerate(spatial_superpixels):
          saliency_map_i     += coefficients[i] * superpixel
          saliency_map_i_abs += abs_coefficients[i] * superpixel

  # Ranking based on the absolute coefficients
  all_superpixels_index = sorted_per_importance_superpixels_index(saliency_map_i_abs)

  # Insertion
  initial_blurred_instance = copy.deepcopy(instance)
  initial_blurred_instance[..., channel_prec] = std_zero_value
  original_prediction = ensemble_predict(models, instance, x3_instance)
  denormalized_H_original_prediction = H_station - (original_prediction * vott_target_test_std + vott_target_test_mean)

  errors_insertion,auc_insertion = insertion(models, instance, x3_instance, all_superpixels_index, initial_blurred_instance, denormalized_H_original_prediction)

  # Deletion
  errors_deletion,auc_deletion  = deletion(models, instance, x3_instance, all_superpixels_index, denormalized_H_original_prediction)

  return saliency_map_i, errors_insertion,auc_insertion, errors_deletion,auc_deletion


### ***Experiments***

""" """

#### ***Cineca***"""

# # LIME combination
# slic_param = [(4,4), (4,5), (4,6), (4,10),(4,20), (4,25), (4,50),
#               (7,4), (7,10),(7,50),
#               (8,2),  (8,15),
#               (9,20)]

# alpha = [0.1, 1.0, 10.0, 100.0]
# percentile_kernel_width = [80, 85, 90, 95]

# # SHAP combination
# slic_param = [(4,4), (4,5), (4,6), (4,10),(4,20), (4,25), (4,50),
#               (7,4), (7,10),(7,50),
#               (8,2),  (8,15),
#               (9,20)]

from sklearn.linear_model import Ridge
import pickle
import numpy as np

# Channels
channel_prec, channel_tmax, channel_tmin = 0, 1, 2
channels = [channel_prec, channel_tmax, channel_tmin]

# Models and dimensions
models = vott_lstm_models_loaded
T, H, W, C = (104, 5, 8, 3)
input_size_spatial = (H, W)
std_zero_value = -0.6486319166678826

# SLIC parameters
slic_param = [(4, 20), (4, 25), (7,4), (4, 10), (7, 10), (8,2), (8,15), (9,20)]
#slic_param = [(8,2)]

# Alpha values and kernel width for LIME
alpha_values = [0.1, 10.0]
#alpha_values = [10.0]
percentile_kernel_width_values = [50, 90]
#percentile_kernel_width_values = [90]

# Length of the test set
len_test_set = len(vottignasco_test_image)

# Main loop over the SLIC parameters
for nr_setup, slic_p in enumerate(slic_param):
    print(f"############################## Setup #{nr_setup}: {slic_p} ##############################")
    n_s, comp = slic_p
    param_combination = f"ns_{n_s}_comp_{comp}"

    # Dictionary to save results
    results = {"lime": {}, "shap": {}}
    print(f"############################## Parameters Combination: {param_combination} ##############################")

    # Loop over the test set instances
    for nr_instance, _ in enumerate(vottignasco_test_image):
        print(f"###################### Explanation for Instance #{nr_instance} ####################################")
        base_start_time_lime_shap = datetime.datetime.now()

        superpixels, zs_primes, perturbed_instances, preds_masked = lime_shap(nr_instance, vottignasco_test_image, vottignasco_test_OHE, channels, models,
                                                                              n_s, comp, input_size_spatial, std_zero_value)
        
        base_end_time_lime_shap = datetime.datetime.now()
        exec_time_base_lime_shap = base_end_time_lime_shap - base_start_time_lime_shap

        nr_coefficients = len(superpixels)
        instance = copy.deepcopy(vottignasco_test_image[nr_instance])

        # Prepare input for the regressor
        X = np.array([z.flatten() for z in zs_primes])  # Masks as rows
        y = np.array(preds_masked)  # Corresponding predictions

        ################# SHAP #################################
        time_start_shap = datetime.datetime.now()

        # SHAP: calculate weights and regression
        M = len(superpixels)
        weights_shap = calculate_weights_shap(M, zs_primes)
        regressor_shap = Ridge(alpha=0.0)
        regressor_shap.fit(X, y, sample_weight=weights_shap)
        coefficients_shap = regressor_shap.coef_

        time_end_shap = datetime.datetime.now()
        exec_time_shap =  (exec_time_base_lime_shap + (time_end_shap - time_start_shap)).total_seconds()

        param_combination_shap = f"ns_{n_s}_comp_{comp}"

        # Insertion and Deletion
        saliency_map_shap_i, errors_insertion_shap, auc_insertion_shap, errors_deletion_shap, auc_deletion_shap = calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients_shap, superpixels)

        # Initialize the dictionary for SHAP only once
        if param_combination_shap not in results["shap"]:
            results["shap"][param_combination_shap] = {
                "coefficients": np.zeros((len_test_set, nr_coefficients)),
                "saliency_maps": np.zeros((len_test_set, H, W)),
                "errors_insertion": np.zeros((len_test_set, nr_coefficients+1)),
                "auc_insertion": np.zeros((len_test_set, 1)),
                "errors_deletion": np.zeros((len_test_set, nr_coefficients+1)),
                "auc_deletion": np.zeros((len_test_set, 1)),
                "executions_times": np.zeros((len_test_set, 1)),
                "parameters_comb": param_combination_shap
        }
        
        # Save the coefficients for each instance
        results["shap"][param_combination_shap]["coefficients"][nr_instance, :] = coefficients_shap
        results["shap"][param_combination_shap]["saliency_maps"][nr_instance, :] = saliency_map_shap_i
        results["shap"][param_combination_shap]["errors_insertion"][nr_instance, :] = errors_insertion_shap
        results["shap"][param_combination_shap]["auc_insertion"][nr_instance, :] = auc_insertion_shap
        results["shap"][param_combination_shap]["errors_deletion"][nr_instance, :] = errors_deletion_shap
        results["shap"][param_combination_shap]["auc_deletion"][nr_instance, :] = auc_deletion_shap
        results["shap"][param_combination_shap]["executions_times"][nr_instance, :] = exec_time_shap
        ########################################## END SHAP ########################################

        ######################################## LIME ############################################
        # LIME: loop over hyperparameters
        for alpha in alpha_values:
            for kernel_width_p in percentile_kernel_width_values:
                time_start_lime = datetime.datetime.now()

                weights_lime = calculate_weights_lime(instance, perturbed_instances, percentile_kernel_width=kernel_width_p)
                regressor_lime = Ridge(alpha=alpha)
                regressor_lime.fit(X, y, sample_weight=weights_lime)
                coefficients_lime = regressor_lime.coef_

                time_end_lime = datetime.datetime.now()
                exec_time_lime = (exec_time_base_lime_shap + (time_end_lime - time_start_lime)).total_seconds()

                param_combination_lime = f"ns_{n_s}_comp_{comp}_kw_{kernel_width_p}_alpha_{alpha}"

                # Calculate Saliency Map, Insertion/Deletion
                saliency_map_lime_i, errors_insertion_lime, auc_insertion_lime, errors_deletion_lime, auc_deletion_lime = calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients_lime, superpixels)

                if param_combination_lime not in results["lime"]:
                    results["lime"][param_combination_lime] = {
                        "coefficients": np.zeros((len_test_set, nr_coefficients)),
                        "saliency_maps": np.zeros((len_test_set, H, W)),
                        "errors_insertion": np.zeros((len_test_set, nr_coefficients+1)),
                        "auc_insertion": np.zeros((len_test_set, 1)),
                        "errors_deletion": np.zeros((len_test_set, nr_coefficients+1)),
                        "auc_deletion": np.zeros((len_test_set, 1)),
                        "executions_times": np.zeros((len_test_set, 1)),
                        "parameters_comb": param_combination_lime
                    }

                # Save the coefficients
                results["lime"][param_combination_lime]["coefficients"][nr_instance, :] = coefficients_lime
                results["lime"][param_combination_lime]["saliency_maps"][nr_instance, :] = saliency_map_lime_i
                results["lime"][param_combination_lime]["errors_insertion"][nr_instance, :] = errors_insertion_lime
                results["lime"][param_combination_lime]["auc_insertion"][nr_instance, :] = auc_insertion_lime
                results["lime"][param_combination_lime]["errors_deletion"][nr_instance, :] = errors_deletion_lime
                results["lime"][param_combination_lime]["auc_deletion"][nr_instance, :] = auc_deletion_lime
                results["lime"][param_combination_lime]["executions_times"][nr_instance, :] = exec_time_lime

    # Save results once for each setup
    path_to_save_results = f"{RESULT_DIR}/###### REPLACE WITH REAL PATH #######/lime_shap_spatial_results_setup_ns_{n_s}_comp_{comp}.pkl"

    with open(path_to_save_results, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved in {path_to_save_results}")
