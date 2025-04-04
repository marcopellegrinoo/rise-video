
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
import matplotlib.pyplot as plt
import sys
# Save Execution Time
import datetime

"""
##### ***Data & Black-Box***

"""

RESULT_DIR = str(sys.argv[1])
print(f"RESULT_DIR: {RESULT_DIR}")

# IMPORTING DATA FOR VOTTIGNASCO
# Get the actual path from an environment variable
work_path = os.environ['WORK']  # Get the value of the WORK environment variable

# Replace these paths with actual paths to the files
v_test_OHE_path = "###### REPLACE WITH REAL PATH #######/Vottignasco/Vottignasco_00425010001_test_month_OHE.npy"
v_test_image_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_normalized_image_sequences.npy"
v_test_target_dates_path = "###### REPLACE WITH REAL PATH #######/Vottignasco/Vottignasco_00425010001_test_target_dates.npy"
v_test_images_dates_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_image_sequences_dates.npy"
shapefile_path = "###### REPLACE WITH REAL PATH #######/Vottignasco/shapefile_raster/"
v_test_normalization_factors_std_path = "###### REPLACE WITH REAL PATH #######/normalization_factors/Vottignasco_00425010001_training_target_std.npy"
v_test_normalization_factors_mean_path = "###### REPLACE WITH REAL PATH #######/normalization_factors/Vottignasco_00425010001_training_target_mean.npy"

# Load the numpy arrays from the files
vottignasco_test_OHE    = np.load(v_test_OHE_path)
vottignasco_test_image  = np.load(v_test_image_path)
vottignasco_test_dates  = np.load(v_test_target_dates_path)
vottignasco_test_images_dates = np.load(v_test_images_dates_path)
vott_target_test_std    = np.load(v_test_normalization_factors_std_path)
vott_target_test_mean   = np.load(v_test_normalization_factors_mean_path)


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


"""### ***LIME and SHAP: Spatio_Temporal***

#### ***Spatial-Temporal Superpixels***

###### ***Temporal***
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import imageio
# from IPython.display import Image

def get_season(day):
  spring = np.arange(80, 172)
  summer = np.arange(172, 264)
  fall = np.arange(264, 355)

  if day in spring:
    season = 'Spring'
  elif day in summer:
    season = 'Summer'
  elif day in fall:
    season = 'Autumn'
  else:
    season = 'Winter'

  return season

# Maps seasons to colors
season_colors = {
    'Winter': 'blue',
    'Spring': 'green',
    'Summer': 'yellow',
    'Autumn': 'orange'
}

def cluster_seasons(seasons):
    clusters = []
    start_index = 0

    for i in range(1, len(seasons)):
        if seasons[i] != seasons[i - 1]:  # Season changes
            clusters.append((start_index, i - 1, seasons[start_index]))  # Save the previous cluster
            start_index = i  # Start a new cluster

    # Add the last cluster
    clusters.append((start_index, len(seasons) - 1, seasons[start_index]))

    return clusters

def create_temporal_superpixels(nr_instance, data_test_image_dates):
  # Convert dates to pandas datetime
  dates = pd.to_datetime(data_test_image_dates[nr_instance])

  # Extract the days and identify the seasons
  tm_days = [date.timetuple().tm_yday for date in dates]
  seasons = [get_season(tm_yday) for tm_yday in tm_days]

  temporal_superpixels = cluster_seasons(seasons)

  return temporal_superpixels

# # Example

# nr_instance = 34

# temporal_superpixels = create_temporal_superpixels(nr_instance, vottignasco_test_images_dates)

# for start, end, season in temporal_superpixels:
#     print(f"Cluster {season}: from index {start} to {end}")

"""###### ***Spatial***"""

import geopandas as gpd
import xarray
import rioxarray
from skimage.segmentation import slic
import matplotlib.pyplot as plt

def create_spatial_superpixels(shapefile_path, n_segments=8, compactness=15):
  # DTM [50m] import
  dtm_piemonte = rioxarray.open_rasterio(shapefile_path + 'DTMPiemonte_filled_50m.tif')
  dtm_piemonte = dtm_piemonte.rio.reproject("epsg:4326")
  dtm_piemonte = dtm_piemonte.where(dtm_piemonte != -99999) # Take valid pixels

  # Catchment shapefile
  catchment = gpd.read_file(shapefile_path + "BAC_01_bacialti.shp") # select GRANA-MAIRA and VARAITA
  catchment = catchment.to_crs('epsg:4326')

  # Select only the Grana-Maira catchment
  catchment_GM = catchment.loc[catchment.NOME == "GRANA-MAIRA"]
  catchment_GM = catchment_GM.reset_index(drop = True)

  # Retrieve the borders of the catchment from the shapefile
  xmin_clip, ymin_clip, xmax_clip, ymax_clip = catchment_GM.total_bounds
  # Extend the borders to include more pixels on the borders

  increase = 0.05 # Degrees
  #ymin_clip -= increase # not needed
  xmin_clip += increase # "+" for subset for pixels included in the mask
  xmax_clip += increase
  #ymax_clip += increase # not needed

  dtm_piemonte_clipped = dtm_piemonte.rio.clip_box( minx = xmin_clip, maxx= xmax_clip , miny= ymin_clip , maxy= ymax_clip)

  # Create img 5x8 with lat, lon, dtm
  # Define the coordinates
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

  for nr_lat,latitude in enumerate(lat):
    for  nr_lon,longitude in enumerate(lon):
      img[nr_lat, nr_lon, 2] = dtm_piemonte_clipped.sel(x=longitude, y=latitude, method='nearest').values

  img = np.nan_to_num(img, nan=0.0)

  # SLIC
  segments = slic(img, n_segments=n_segments, compactness=compactness)

  # Create Spatial-Superpixels
  # Find unique values in the matrix (clusters)
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


"""#### ***Generation and Application of Uniform Masks (3D)***"""

import numpy as np
from tqdm import tqdm

def generate_masks(spatial_superpixel_clusters, temporal_superpixels, shape):
    time_steps, height, width, channels = shape
    masks = []

    for t_sp in tqdm(temporal_superpixels, desc='Generating masks'):
        start, end, _ = t_sp

        for cluster in spatial_superpixel_clusters:
            mask = np.ones((time_steps, height, width))  # All pixels set to 1
            for h, w in cluster:
                mask[start:end+1, h, w] = 0  # Space-time cluster pixels set to 0

            masks.append(mask)

    return np.array(masks)

"""#### ***Interpretable Representations***"""

import numpy as np

def generate_contiguous_ones(length, n):
    results = []

    # Iterate through the list with a window of size n
    for i in range(length - n + 1):
        # Create a zero vector
        vec = np.zeros(length, dtype=int)
        # Set the elements within the window [i:i+n] to 1
        vec[i:i + n] = 1
        # Add the vector to the results list
        results.append(vec)

    return np.array(results)

import itertools

# Creates vectors with a single 0 for each element
def create_zs_each_superpixel_foreach_season(n):
  zs_primes = []
  for i in range(n):
      # Create a zero vector
      vec = np.ones(n, dtype=int)
      # Set a single element to 1
      vec[i] = 0
      # Add the vector to the permutations list
      zs_primes.append(vec)
  return zs_primes

# Creates zs_primes with all 0 for each season
def create_zs_season(masks, spatial_superpixels):
  zs_primes = []
  len_masks = len(masks)
  nr_spatial_superpixel = len(spatial_superpixels)

  # Iterate from 0 to len_masks in steps of nr_spatial_superpixel
  for i in range(0, len_masks, nr_spatial_superpixel):
    # Create a vector of 1s
    vec = np.ones(len_masks, dtype=int)
    # Set elements in the range [i : i + nr_spatial_superpixel] to 0, with boundary checks
    vec[i:i + nr_spatial_superpixel] = 0
    # Add the resulting vector to the list
    zs_primes.append(vec)
  return zs_primes

def create_zs_superpixel_foreach_season(masks, spatial_superpixels, temporal_superpixels, seasons_to_perturb):
    """ Perturb the superpixels for each season based on the passed binary vectors. """
    zs_primes = []
    len_masks = len(masks)
    nr_spatial_superpixel = len(spatial_superpixels)  # Number of spatial superpixels
    nr_temporal_superpixel = len(temporal_superpixels)  # Number of temporal superpixels

    for i in range(nr_spatial_superpixel):  # Loop over each spatial superpixel
        vec = np.ones(len_masks, dtype=int)  # Initialize the vector with all 1s
        for j, s in enumerate(seasons_to_perturb):  # Loop over the seasons to perturb
            if s == 1:  # If the season should be perturbed
                # Modify the value in the vector for the current temporal superpixel
                # Correctly calculate the temporal season index
                temporal_index = nr_spatial_superpixel * j + i
                if temporal_index < len_masks:  # Check if the temporal index is valid
                    vec[temporal_index] = 0

        zs_primes.append(vec)  # Add the perturbed vector to the list

    return zs_primes
def create_zs_superpixel_for_contiguous_frame(masks, spatial_superpixels, temporal_superpixels):
    """ Creates a list of vectors to perturb the superpixels during the seasons
        with contiguous blocks of 1s.
    """
    zs_prime = []  # List that will contain the final results
    nr_temporal_superpixel = len(temporal_superpixels)

    # Loop through the various n (number of seasons to perturb)
    for n in range(0, nr_temporal_superpixel):
        binary_vectors = generate_contiguous_ones(nr_temporal_superpixel, n)  # Vectors with contiguous blocks of 1s

        # Loop over each generated binary vector
        for bv in binary_vectors:
            #print("Binary Vector:", bv)
            vecs = create_zs_superpixel_foreach_season(masks, spatial_superpixels, temporal_superpixels, bv)
            zs_prime.append(vecs)
            #print("Perturbed Vectors:", vecs)
            #print("\n ----------- \n")

    # Flatten the list of lists into a single list
    zs_prime = list(itertools.chain.from_iterable(zs_prime))
    return zs_prime


import numpy as np

def generate_random_zs_primes(N, length, seed, prob_activation=0.1):
    """
    Generates N binary vectors (zs_primes) with activated masks with probability p.

    Args:
        N (int): Number of vectors to generate.
        length (int): Length of each vector.
        prob_activation (float): Probability of activating a mask (default 0.1).

    Returns:
        list: List of NumPy arrays with the generated vectors.
    """

    np.random.seed(seed)

    zs_primes = []
    for _ in range(N):
        vec = np.random.choice([1, 0], size=length, p=[1-prob_activation, prob_activation])
        zs_primes.append(vec)
    return zs_primes

"""#### ***Application Masks***"""

def multiplicative_uniform_noise_onechannel(instance, zs_primes, masks, channel, std_zero_value=-0.6486319166678826):
  """
  param:masks: masks generated for each superpixel
  """
  masked = []
  for z in zs_primes:
    masked_instance = copy.deepcopy(instance)
    for i,z_i in enumerate(z):
      if z_i == 0:
         # Apply the perturbation only to the specified channel
        masked_instance[..., channel] = (
            masked_instance[..., channel] * masks[i] + (1 - masks[i]) * std_zero_value)

    masked.append(masked_instance)

  return masked

"""#### ***Prediction on Perturbed Instances***"""

import tensorflow as tf
import numpy as np

def ensemble_predict(models, images, x3_exp, batch_size=1000):
    # Make sure images is a list
    if not isinstance(images, list):
        images = [images]

    len_x3 = len(images)

    # Convert x3_exp to a tensor replicated for each image
    x3_exp_tensor = tf.convert_to_tensor(x3_exp, dtype=tf.float32)

    # List to collect the final predictions
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

        # Convert the batch predictions to a tensor and compute the mean
        batch_preds_tensor = tf.stack(batch_preds)
        mean_batch_preds = tf.reduce_mean(batch_preds_tensor, axis=0)

        # Add the batch predictions to the final list
        final_preds.extend(mean_batch_preds.numpy())

    return np.array(final_preds)

"""#### ***Regressor Weights Calculation***

###### ***LIME***
Where *calculate_D*:
* D is the L2-Distance (Euclidean Distance)
* x is the original instance to explain
* z is the perturbed, non-interpretable version
"""

def calculate_D(instance, perturbed_instance):
  x = instance.flatten()
  z = perturbed_instance.flatten()

  return np.linalg.norm(x - z)

def calculate_weights_lime(instance, perturbed_instances, percentile_kernel_width):
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
    Calculates the Kernel SHAP weight for a given mask (interpretable instance).

    Args:
        M (int): Total number of features.
        z (array): Array containing a zs_prime.

    Returns:
        float: Kernel weight value for z'.
    """

  z_size = np.sum(z)
  #print("Mask size: ", mask_size)
  if z_size == 0 or z_size == M:
    return 0  # Null weight in these extreme cases
  # Binomial coefficient: M choose subset_size (|z'|)
  # Kernel SHAP weight formula
  weight = (M-1)/(binom(M, z_size)*(z_size*(M-z_size)))
  return weight

def calculate_weights_shap(M, zs_primes):
  weights = []

  for z in zs_primes:
    w = shap_kernel_weight(M, z)
    weights.append(w)

  weights = np.array(weights)
  return weights

"""#### ***Lime-Shap Spatio_Temporal: Framework***"""

from sklearn.linear_model import Ridge

def lime_shap_st(nr_instance, dataset_test_image, dataset_test_OHE, data_test_image_dates, channels, models,
                 spatial_superpixels, spatial_superpixels_clusters,
                 N, n_segments, compactness, input_size, H_station=390.0, seed=42, std_zero_value=-0.6486319166678826):

  """
  param:int input_size temporal dimension of the data (104 in our case)
  """

  channel_prec, channel_tmax, channel_min = channels
  instance    = copy.deepcopy(dataset_test_image[nr_instance])  # instance to explain
  x3_instance = copy.deepcopy(dataset_test_OHE[nr_instance])    # One-Hot encode months of the instance's frames

  shape = input_size

  # Temporal Superpixels
  temporal_superpixels = create_temporal_superpixels(nr_instance, data_test_image_dates)
  # Spatial Superpixels
  #spatial_superpixels, spatial_superpixels_clusters, segments  = create_spatial_superpixels(shapefile_path, n_segments=n_segments, compactness=compactness)

  # Generate masks for creating Neighbors
  masks = generate_masks(spatial_superpixels_clusters, temporal_superpixels, shape)

  # Create Interpretable Representations of Neighbors
  length = len(masks)
  zs_primes = []

  zs_primes.append(create_zs_each_superpixel_foreach_season(len(masks)))
  zs_primes.append(create_zs_season(masks, spatial_superpixels))
  zs_primes.append(create_zs_superpixel_for_contiguous_frame(masks, spatial_superpixels, temporal_superpixels))
  zs_primes.append(generate_random_zs_primes(N, length, seed))

  zs_primes = [item for sublist in zs_primes for item in sublist]

  zs_prime_array = np.array(zs_primes)
  zs_primes = np.unique(zs_prime_array, axis=0)
  # Remove zs with all 1s or 0s
  zs_primes = [sub_array for sub_array in zs_primes if not (all(x == 1 for x in sub_array) or all(x == 0 for x in sub_array))]

  # Create perturbed instances (neighbors)
  perturbed_instances = multiplicative_uniform_noise_onechannel(instance, zs_primes, masks, channel_prec)

  # Predict perturbed instances
  preds_masked = ensemble_predict(models, list(perturbed_instances), x3_instance)
  # Denormalize with respect to the black-box output
  denorm_preds_masked  = [pred_masked * vott_target_test_std + vott_target_test_mean for pred_masked in preds_masked]
  denormalized_H_preds_masked  = [H_station - denorm_pred_masked for denorm_pred_masked in denorm_preds_masked]

  return temporal_superpixels, zs_primes, perturbed_instances, denormalized_H_preds_masked


"""#### ***Evaluation Metrics***"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def calculate_auc(x, y):
    """
    Calculates the area under the curve (AUC) using the trapezoidal rule.

    :param x: x-axis values (fraction of pixels/frames inserted).
    :param y: y-axis values (calculated errors).
    :return: Area under the curve.
    """
    return np.trapz(y, x)

def calculate_auc_and_mean_errors(errors_all_dateset):
  mean_errors = np.mean(errors_all_dateset, axis=0)
  # x array for the number of superpixels inserted
  x = np.arange(0, len(mean_errors))  # Dynamic array based on the data length
  auc = calculate_auc(x, mean_errors)

  return auc, mean_errors

"""##### ***Insertion***"""

def update_instance_with_superpixels(current_instance, original_instance, start, end, list_of_pixel):
    """
    Updates the image by inserting the most important pixels.

    :param current_instance: Current instance.
    :param original_instance: Original instance.
    :param index_of_superpixels: List containing the indices of the considered superpixel.
    :return: Instance updated with the superpixel.
    """
    new_current_instance = current_instance.copy()

    for x, y in list_of_pixel:
        for t in range(start, end):
            new_current_instance[t, x, y, 0] = original_instance[t, x, y, 0]
    return new_current_instance

def insertion(models, original_instance, x3, sorted_per_importance_all_superpixels_index, initial_blurred_instance, original_prediction, H_station=390.0):
    """
    Calculates the insertion metric for a given explanation.

    :param models: List of pre-trained models.
    :param original_instance: Original instance.
    :param x3: One-hot encoding for the prediction.
    :param sorted_per_importance_all_superpixels_index: List of lists of all superpixels sorted by importance.
    :param initial_blurred_images: Initial image with all pixels set to zero.
    :return: List of errors at each insertion step.
    """

    # List to store the instances with pixels added gradually. Initialized with the blurred initial instance
    insertion_images = [initial_blurred_instance]

    # Prediction on the initial image (all pixels set to zero)
    I_prime = copy.deepcopy(initial_blurred_instance)

    # Gradually add the most important pixels (for each frame). I get a list with all images having pixels added gradually
    for start, end, list_of_pixel in sorted_per_importance_all_superpixels_index:
        I_prime = update_instance_with_superpixels(I_prime, original_instance, start, end, list_of_pixel)
        insertion_images.append(I_prime)

    # Calculate the predictions for the instances where pixels were added gradually
    new_predictions = ensemble_predict(models, insertion_images, x3)
    denorm_new_predictions = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    
    # For each of these predictions, calculate the MSE relative to the prediction on the original instance (as in the test set).
    # Ignore the first one which is on the original blurred image.
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions[1:]]

    initial_error = mean_squared_error(original_prediction, denormalized_H_new_predictions[0])
    print(f"Initial Prediction with Blurred Instance, new prediction: {denormalized_H_new_predictions[0]}, error: {initial_error}")
    only_inserted_pixel_new_predictions = denormalized_H_new_predictions[1:]

    for nr_superpixel, error in enumerate(errors):
        print(f"SuperPixel: {sorted_per_importance_all_superpixels_index[nr_superpixel]}, new prediction: {only_inserted_pixel_new_predictions[nr_superpixel]}, error: {error}")

    total_errors = [initial_error] + errors  # Initial error + errors for all inserted pixels

    # New x-axis: number of superpixels inserted (1, 2, ..., 8)
    x = np.arange(0, len(total_errors))  # From 0 to 8 inclusive
    #print(x)

    x_for_auc = np.linspace(0, 1, len(total_errors))
    # Calculate the AUC with the new x-axis
    auc = calculate_auc(x_for_auc, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    return total_errors, auc

"""##### ***Deletion***"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def update_instance_removing_superpixels(current_instance, start, end, list_of_pixel, std_zero_value=-0.6486319166678826):
    """
    Updates the image by removing the most important pixels.

    :param current_instance: Current instance.
    :param original_instance: Original instance.
    :param index_of_superpixels: List containing the indices of the considered superpixel.
    :return: Instance updated with the superpixel.
    """
    new_current_instance = current_instance.copy()
    for x, y in list_of_pixel:
        for t in range(start, end):
            new_current_instance[t, x, y, 0] = std_zero_value
    return new_current_instance

def deletion(models, original_instance, x3_instance, sorted_per_importance_all_superpixels_index, original_prediction, H_station=390.0):
    """
    Calculates the deletion metric for a given explanation.

    :param models: List of pre-trained models.
    :param original_instance: Original instance.
    :param x3_instance: One-hot encoding for the prediction.
    :param sorted_per_importance_all_superpixels_index: List of lists of all superpixels sorted by importance.
    :param original_prediction: Original prediction.
    :return: List of errors at each deletion step.
    """

    # List to store the instances where pixels are gradually removed. Initialized with the original instance.
    deletion_images = []

    # Prediction on the initial image (all pixels set to zero)
    I_prime = copy.deepcopy(original_instance)

    # Gradually remove the most important pixels (for each frame). I get a list with all images having pixels removed gradually
    for start, end, list_of_pixel in sorted_per_importance_all_superpixels_index:
        I_prime = update_instance_removing_superpixels(I_prime, start, end, list_of_pixel)
        deletion_images.append(I_prime)

    # Calculate predictions on all images where pixels were gradually removed
    new_predictions = ensemble_predict(models, deletion_images, x3_instance)
    denorm_new_predictions = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    
    # Calculate MSE with respect to the original prediction
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions]

    initial_error = 0.0
    print(f"Initial Prediction with Original instance, prediction: {original_prediction}, error: {initial_error}")

    for nr_superpixel, error in enumerate(errors):
        print(f"Removed SuperPixel: {sorted_per_importance_all_superpixels_index[nr_superpixel]}, new prediction: {denormalized_H_new_predictions[nr_superpixel]}, error: {error}")

    total_errors = [initial_error] + errors  # Initial error + errors for all removed pixels

    # Plot
    # New x-axis: number of superpixels inserted (1, 2, ..., 8)
    x = np.arange(0, len(total_errors))  # From 0 to 8 inclusive
    #print(x)
    x_for_auc = np.linspace(0, 1, len(total_errors))
    # Calculate the AUC with the new x-axis
    auc = calculate_auc(x_for_auc, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    return total_errors, auc


"""### ***Experiments***"""

def create_frame_for_saliency_video(shape, coefficients, spatial_superpixels, height=5, width=8):
  """
  Args:
   - shape: (time_steps, height, width, nr_channels)
   - coefficients: spatial_superpixel coefficients for temporal cluster
   - spatial_superpixels: 5x8 matrices for spatial superpixels
  """

  frame = np.zeros((height, width))

  for i, superpixel in enumerate(spatial_superpixels):
    frame += superpixel * coefficients[i]

  return frame

def find_top_indices(matrix):
    # Flatten the matrix and sort the indices based on descending values
    flat_indices = np.argsort(matrix.flatten())[::-1]
    # Convert the "flat" indices to coordinates (x, y)
    indices = [np.unravel_index(idx, matrix.shape) for idx in flat_indices]
    return indices

def calculate_saliency_video_insertion_deletion_errors_auc(nr_instance, coefficients, temporal_superpixels, spatial_superpixels, spatial_superpixels_clusters, nr_temporal_superpixel, nr_spatial_superpixel,
                                                           models=vott_lstm_models_loaded, H_station =390.0, channel_prec=0, std_zero_value=-0.6486319166678826, input_size=(104,5,8,3), T=104, H=5, W=8):
  coefficients_reshape     = coefficients.reshape(nr_temporal_superpixel, nr_spatial_superpixel)
  abs_coefficients_reshape = np.abs(coefficients).reshape(nr_temporal_superpixel, nr_spatial_superpixel)

  saliency_video_i     = np.zeros((T, H, W))
  saliency_video_i_abs = np.zeros((T, H, W))

  # Create all frames for each identified season for the Saliency Video
  frames_for_t_superpixels     = [create_frame_for_saliency_video(input_size, coeff, spatial_superpixels) for coeff in coefficients_reshape]
  frames_for_t_superpixels_abs = [create_frame_for_saliency_video(input_size, coeff, spatial_superpixels) for coeff in abs_coefficients_reshape]

  for i, t_superpixel in enumerate(temporal_superpixels):
      start, end, _ = t_superpixel
      saliency_video_i[start:end+1] = frames_for_t_superpixels[i]
      saliency_video_i_abs[start:end+1] = frames_for_t_superpixels_abs[i]

  # Ranking with the absolute values of the coefficients
  # ST-Superpixels ordered by importance
  superpixels_importance_cluster = find_top_indices(abs_coefficients_reshape)
  sorted_per_importance_all_superpixels_index = []
  for nr_ts, nr_ss in superpixels_importance_cluster:
    start, end = temporal_superpixels[nr_ts][0], temporal_superpixels[nr_ts][1] + 1
    cluster_spatial_superpixel = spatial_superpixels_clusters[nr_ss]
    sorted_per_importance_all_superpixels_index.append((start, end, cluster_spatial_superpixel))

  # Insertion
  instance    = copy.deepcopy(vottignasco_test_image[nr_instance])
  x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])    # One-Hot encode months of the instance frames

  all_superpixels_index = sorted_per_importance_all_superpixels_index
  initial_blurred_instance = copy.deepcopy(instance)
  initial_blurred_instance[:,:,:,channel_prec] = std_zero_value

  original_prediction = ensemble_predict(models, instance, x3_instance)
  denormalized_H_original_prediction = H_station - (original_prediction * vott_target_test_std + vott_target_test_mean)

  errors_insertion, auc_insertion = insertion(models, instance, x3_instance, all_superpixels_index, initial_blurred_instance, denormalized_H_original_prediction)

  # Deletion
  errors_deletion, auc_deletion = deletion(models, instance, x3_instance, all_superpixels_index, denormalized_H_original_prediction)

  return saliency_video_i, errors_insertion, auc_insertion, errors_deletion, auc_deletion

"""" Experiments """

from sklearn.linear_model import Ridge
import pickle
import numpy as np

# Channels
channel_prec, channel_tmax, channel_tmin = 0, 1, 2
channels = [channel_prec, channel_tmax, channel_tmin]

# Models and sizes
models = vott_lstm_models_loaded
T, H, W, C = (104, 5, 8, 3)
input_size = (T, H, W, C)
std_zero_value = -0.6486319166678826

N = 2500
seed = 42

# SLIC parameters
slic_param = [(4, 20), (4, 25), (7, 4), (4, 10), (7, 10), (8, 2), (8, 15), (9, 20)]
# Alpha values and kernel width for LIME
alpha_values = [0.1, 10.0]
percentile_kernel_width_values = [50, 90]

# Length of the test set
len_test_set = len(vottignasco_test_image)

# Main loop for SLIC parameters
for nr_setup, slic_p in enumerate(slic_param):
  n_s, comp = slic_p
  param_combination = f"ns_{n_s}_comp_{comp}"

  # Dictionary to save results
  results = {"lime": {}, "shap": {}}

  # Spatial Superpixels
  spatial_superpixels, spatial_superpixels_clusters, segments  = create_spatial_superpixels(shapefile_path, n_segments=n_s, compactness=comp)

  print(f"############################## Parameters Combination: {param_combination} ##############################")

  # Loop through the test set instances
  for nr_instance, _ in enumerate(vottignasco_test_image):
    print(f"############################## ST LIME and SHAP for instance #{nr_instance} with {param_combination} ##############################")
    instance = copy.deepcopy(vottignasco_test_image[nr_instance])
    x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])    # One-Hot encode months of the instance frames

    base_start_time_lime_shap = datetime.datetime.now()

    temporal_superpixels, zs_primes, perturbed_instances, preds_masked = lime_shap_st(nr_instance, vottignasco_test_image, vottignasco_test_OHE, vottignasco_test_images_dates, channels, models,
                                                                                      spatial_superpixels, spatial_superpixels_clusters,
                                                                                      N, n_s, comp, input_size, seed=seed)
    
    base_end_time_lime_shap = datetime.datetime.now()
    exec_time_base_lime_shap = base_end_time_lime_shap - base_start_time_lime_shap
    
    nr_temporal_superpixel = len(temporal_superpixels)
    nr_spatial_superpixel  = len(spatial_superpixels)
    nr_coefficients = nr_spatial_superpixel * nr_temporal_superpixel

    # Prepare input for the regressor
    X = np.array([z.flatten() for z in zs_primes])  # Masks as rows
    y = np.array(preds_masked)  # Corresponding predictions

    ################################## SHAP ###################################
    time_start_shap = datetime.datetime.now()
    # SHAP: calculate the weights and regression
    M = nr_coefficients
    weights_shap = calculate_weights_shap(M, zs_primes)
    regressor_shap = Ridge(alpha=0.0)
    regressor_shap.fit(X, y, sample_weight=weights_shap)
    coefficients_shap = regressor_shap.coef_

    time_end_shap = datetime.datetime.now()
    exec_time_shap =  (exec_time_base_lime_shap + (time_end_shap - time_start_shap)).total_seconds()

    # Insertion/Deletion
    saliency_video_i_shap, errors_insertion_shap, auc_insertion_shap, errors_deletion_shap, auc_deletion_shap = calculate_saliency_video_insertion_deletion_errors_auc(nr_instance, coefficients_shap, temporal_superpixels, spatial_superpixels, spatial_superpixels_clusters,
                                                                                                                                                                    nr_temporal_superpixel, nr_spatial_superpixel, models, channel_prec=channel_prec)

    param_combination_shap = f"ns_{n_s}_comp_{comp}"

    # Initialize the dictionary for SHAP only once
    if param_combination_shap not in results["shap"]:
      results["shap"][param_combination_shap] = {
              "coefficients": [None] * len_test_set,  # List to support variable lengths,
              "saliency_videos": np.zeros((len_test_set, T, H, W)),
              "errors_insertion": [None] * len_test_set,
              "auc_insertion": np.zeros((len_test_set, 1)),
              "errors_deletion": [None] * len_test_set,
              "auc_deletion": np.zeros((len_test_set, 1)),
              "executions_times": np.zeros((len_test_set, 1)),
              "parameters_comb": param_combination_shap
            }

    # Save coefficients
    results["shap"][param_combination_shap]["coefficients"][nr_instance]        = coefficients_shap
    results["shap"][param_combination_shap]["saliency_videos"][nr_instance, :]  = saliency_video_i_shap
    results["shap"][param_combination_shap]["errors_insertion"][nr_instance]    = errors_insertion_shap
    results["shap"][param_combination_shap]["auc_insertion"][nr_instance, :]    = auc_insertion_shap
    results["shap"][param_combination_shap]["errors_deletion"][nr_instance]     = errors_deletion_shap
    results["shap"][param_combination_shap]["auc_deletion"][nr_instance, :]     = auc_deletion_shap
    results["shap"][param_combination_shap]["executions_times"][nr_instance, :] = exec_time_shap
    ###################################### END SHAP ####################################

    ######################## LIME ################################################
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

        # Insertion/Deletion
        saliency_video_i_lime, errors_insertion_lime, auc_insertion_lime, errors_deletion_lime, auc_deletion_lime = calculate_saliency_video_insertion_deletion_errors_auc(nr_instance, coefficients_lime, temporal_superpixels, spatial_superpixels, spatial_superpixels_clusters,
                                                                                                                                                                        nr_temporal_superpixel, nr_spatial_superpixel, 
                                                                                                                                                                        models, channel_prec=channel_prec)
        param_combination_lime = f"ns_{n_s}_comp_{comp}_kw_{kernel_width_p}_alpha_{alpha}"

        # Initialize the dictionary for LIME only once
        if param_combination_lime not in results["lime"]:
              results["lime"][param_combination_lime] = {
              "coefficients": [None] * len_test_set,  # List to support variable lengths,
              "saliency_videos": np.zeros((len_test_set, T, H, W)),
              "errors_insertion": [None] * len_test_set,
              "auc_insertion": np.zeros((len_test_set, 1)),
              "errors_deletion": [None] * len_test_set,
              "auc_deletion": np.zeros((len_test_set, 1)),
              "executions_times": np.zeros((len_test_set, 1)),
              "parameters_comb": param_combination_lime
            }

        # Save coefficients
        results["lime"][param_combination_lime]["coefficients"][nr_instance]        = coefficients_lime
        results["lime"][param_combination_lime]["saliency_videos"][nr_instance, :]  = saliency_video_i_lime
        results["lime"][param_combination_lime]["errors_insertion"][nr_instance]    = errors_insertion_lime
        results["lime"][param_combination_lime]["auc_insertion"][nr_instance, :]    = auc_insertion_lime
        results["lime"][param_combination_lime]["errors_deletion"][nr_instance]     = errors_deletion_lime
        results["lime"][param_combination_lime]["auc_deletion"][nr_instance, :]     = auc_deletion_lime
        results["lime"][param_combination_lime]["executions_times"][nr_instance, :] = exec_time_lime
        ##################################### END LIME #################################################

  print(f"################################ END all dataset for {param_combination} ################################ ")
  # Save results once for each setup
  #path_to_save_results = os.path.join(work_path, f"Water_Resources/rise-video/XAI/spatial_temporal/results/lime_shap_multiplicative_norm_zero/corrected_ins_del_lime_shap_st_results_setup_ns_{n_s}_comp_{comp}.pkl") 
  path_to_save_results = f"{RESULT_DIR}/lime_shap_st_results_setup_ns_{n_s}_comp_{comp}.pkl"

  with open(path_to_save_results, 'wb') as f:
    pickle.dump(results, f)
