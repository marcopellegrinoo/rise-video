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

# Vottignasco Data

work_path = os.environ['WORK'] 
v_test_OHE_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_month_OHE.npy")
v_test_image_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_normalized_image_sequences.npy")
v_test_target_dates_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_target_dates.npy")
v_test_images_dates_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_image_sequences_dates.npy")
v_test_normalization_factors_std_path = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/normalization_factors/Vottignasco_00425010001_training_target_std.npy")
v_test_normalization_factors_mean_path     = os.path.join(work_path, "###### REPLACE WITH REAL PATH #######/normalization_factors/Vottignasco_00425010001_training_target_mean.npy")

# Load numpy file
vottignasco_test_OHE    = np.load(v_test_OHE_path)
vottignasco_test_image  = np.load(v_test_image_path)
vottignasco_test_dates  = np.load(v_test_target_dates_path)
vottignasco_test_images_dates = np.load(v_test_images_dates_path)
vott_target_test_std    = np.load(v_test_normalization_factors_std_path) 
vott_target_test_mean   = np.load(v_test_normalization_factors_mean_path)


print(len(vottignasco_test_dates))
print(len(vottignasco_test_image))
print(len(vottignasco_test_OHE))
print(len(vottignasco_test_images_dates))

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

"""### ***LIME and SHAP: Temporal***

#### ***Temporal-Superpixels***
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Map seasons to colors
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
        if seasons[i] != seasons[i - 1]:  # Season change
            clusters.append((start_index, i - 1, seasons[start_index]))  # Save the previous cluster
            start_index = i  # Start a new cluster

    # Add the last cluster
    clusters.append((start_index, len(seasons) - 1, seasons[start_index]))

    return clusters

def create_temporal_superpixels(nr_instance, data_test_image_dates):
  # Convert dates to pandas datetime format
  dates = pd.to_datetime(data_test_image_dates[nr_instance])

  # Extract the day of the year and identify seasons
  tm_days = [date.timetuple().tm_yday for date in dates]
  seasons = [get_season(tm_yday) for tm_yday in tm_days]

  temporal_superpixels = cluster_seasons(seasons)

  return temporal_superpixels

"""#### ***Interpretable Representations***"""

# Creation of z'

import itertools
import numpy as np

def generate_z_primes(n):
  """Args
      n (int): number of superpixels considered
     Return
      np.array: all possible combinations of 0s and 1s of length n
  """

  # Generate all possible combinations of 0s and 1s of length n
  z_primes = list(itertools.product([0, 1], repeat=n))
  # Optionally convert tuples to numpy array
  z_primes = np.array(z_primes)
  # Remove the first element (all zeros)
  z_primes = z_primes[1:]
  # Remove the last element (all ones)
  z_primes = z_primes[:-1]
  return z_primes


"""#### ***Neighbours***

##### ***Uniform Noise Mask Generation (1D)***
"""

# temporal_superpixels

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_masks(superpixels, input_size):
    N = len(superpixels)
    masks = np.empty((N, input_size))

    for i, superpixel in tqdm(enumerate(superpixels), desc='Generating masks'):
        start, end, _ = superpixel
        mask = np.ones(input_size)  # All pixels set to 1
        mask[start:end+1] = 0  # Pixels of the cluster set to 0
        masks[i] = mask

    return masks

"""##### ***Application of Masks***"""

def multiplicative_uniform_noise_onechannel(instance, zs_primes, masks, channel, std_zero_value=-0.6486319166678826):
  """
  param: masks: masks generated for each superpixel
  """
  T = instance.shape[0]

  masked = []

  for z in zs_primes:
    masked_instance = copy.deepcopy(instance)
    for i, z_i in enumerate(z):
      if z_i == 0:
        for t in range(T):
          # Apply perturbation only to the specified channel
          masked_instance[t,:,:,channel] = (
              masked_instance[t,:,:,channel] * masks[i][t] + (1 - masks[i][t]) * std_zero_value)

    masked.append(masked_instance)

  return masked


"""#### ***Prediction with Black-Box***"""

import tensorflow as tf
import numpy as np

def ensemble_predict(models, images, x3_exp, batch_size=1000):
    # Make sure images is a list
    if not isinstance(images, list):
        images = [images]

    len_x3 = len(images)

    # Convert x3_exp into a tensor replicated for each image
    x3_exp_tensor = tf.convert_to_tensor(x3_exp, dtype=tf.float32)

    # List to collect final predictions
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

        # Convert batch predictions to tensor and compute mean
        batch_preds_tensor = tf.stack(batch_preds)
        mean_batch_preds = tf.reduce_mean(batch_preds_tensor, axis=0)

        # Add batch predictions to final list
        final_preds.extend(mean_batch_preds.numpy())

    return np.array(final_preds)


"""#### ***Regressor Weight Calculation***

###### ***LIME***
Where *calculate_D*:
* D is the L2-Distance (Euclidean Distance)
* x is the original instance to be explained
* z is the perturbed non-interpretable version
"""

def calculate_D(instance, perturbed_istance):
  x = instance.flatten()
  z = perturbed_istance.flatten()

  return np.linalg.norm(x - z)

def calculate_weigths_lime(instance, perturbed_instances, percentile_kernel_width):
  distances = [calculate_D(instance, perturbed_instance) for perturbed_instance in perturbed_instances]
  kernel_width = np.percentile(distances, percentile_kernel_width)
  # Neighbor importance
  weights = np.exp(- (np.array(distances) ** 2) / (kernel_width ** 2))
  return weights

"""##### ***Kernel-SHAP***"""

import math
from scipy.special import binom

def shap_kernel_weight(M, z):
  """
    Computes the kernel weight for Kernel SHAP for a given interpretable instance (mask).

    Args:
        M (int): Total number of features.
        z (array): Array containing a zs_prime (interpretable binary mask).

    Returns:
        float: Weight value according to the SHAP kernel.
    """

  z_size = np.sum(z)
  #print("Mask size: ", mask_size)
  if z_size == 0 or z_size == M:
    return 0  # Zero weight in these edge cases
  # Binomial coefficient: M choose |z'|
  # Kernel SHAP weight formula
  weight = (M-1)/(binom(M, z_size)*(z_size*(M-z_size)))
  return weight

def calculate_weigths_shap(M, zs_primes):
  weights = []

  for z in zs_primes:
    w = shap_kernel_weight(M, z)
    weights.append(w)

  weights = np.array(weights)
  return weights


"""#### ***Lime-Spatial: Framework***"""

from sklearn.linear_model import Ridge

def lime_shap(nr_instance, dataset_test_image, dataset_test_OHE, dataset_test_images_dates, channels, models,
              input_size, H_station=390.0, std_zero_value=-0.6486319166678826):

  """
  param: int input_size - temporal dimension of the data (104 in our case)
  """

  channel_prec, channel_tmax, channel_min = channels

  instance    = copy.deepcopy(dataset_test_image[nr_instance])  # instance to explain
  x3_instance = copy.deepcopy(dataset_test_OHE[nr_instance])    # One-Hot encoded months corresponding to the instance frames

  # Temporal Superpixel creation
  superpixels = create_temporal_superpixels(nr_instance, dataset_test_images_dates)
  # Interpretable representations of the instance
  zs_primes = generate_z_primes(len(superpixels))
  # Mask generation
  masks = generate_masks(superpixels, input_size)
  # Neighbor creation
  perturbed_instances = multiplicative_uniform_noise_onechannel(instance, zs_primes, masks, channel_prec, std_zero_value)
  # Predictions on perturbed instances
  preds_masked = ensemble_predict(models, list(perturbed_instances), x3_instance, batch_size=1000)
  # Denormalization with respect to the black-box output
  denorm_preds_masked  = [pred_masked * vott_target_test_std + vott_target_test_mean for pred_masked in preds_masked]
  denormalized_H_preds_masked  = [H_station - denorm_pred_masked for denorm_pred_masked in denorm_preds_masked]

  return superpixels, zs_primes, perturbed_instances, denormalized_H_preds_masked


#### ***Evaluation Metrics***

def calculate_auc(x, y):
    """
    Computes the Area Under the Curve (AUC) using the trapezoidal rule.

    :param x: Values on the x-axis (fraction of pixels/frames inserted).
    :param y: Values on the y-axis (calculated errors).
    :return: Area under the curve.
    """
    return np.trapz(y, x)

import numpy as np

def sorted_per_importance_superpixels_index_1D(array):
    """
    Sorts superpixel indices by importance.

    :param array: 1D array with importance scores.
    :return: List of index groups sorted by decreasing importance.
    """
    array = np.array(array)
    unique_values = np.unique(array)

    index_by_value = {val: [] for val in unique_values}
    for idx, val in enumerate(array):
        index_by_value[val].append(idx)

    sorted_values = sorted(index_by_value.keys(), reverse=True)
    results = [index_by_value[val] for val in sorted_values]

    return results

##### ***Insertion***

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def update_instance_with_superpixels(current_instance, original_instance, index_of_superpixels):
    """
    Updates the instance by inserting the most important pixels.

    :param current_instance: Current blurred instance.
    :param original_instance: Original instance.
    :param index_of_superpixels: List of indices for the current superpixel.
    :return: Updated instance with the superpixel inserted.
    """
    new_current_instance = current_instance.copy()

    for t in index_of_superpixels:
        new_current_instance[t, :, :, 0] = original_instance[t, :, :, 0]
    return new_current_instance

def insertion(models, original_instance, x3, sorted_per_importance_all_superpixels_index, initial_blurred_instance, original_prediction, H_station=390.0):
    """
    Computes the Insertion metric for a given explanation.

    :param models: List of pretrained models.
    :param original_instance: Original unperturbed instance.
    :param x3: One-hot encoding for prediction.
    :param sorted_per_importance_all_superpixels_index: List of superpixel indices sorted by importance.
    :param initial_blurred_instance: Initial blurred instance (e.g., all-zero or noisy).
    :param original_prediction: Prediction on the original instance.
    :return: List of errors at each insertion step and the AUC.
    """
    insertion_images = [initial_blurred_instance]
    I_prime = copy.deepcopy(initial_blurred_instance)

    for index_of_superpixels in sorted_per_importance_all_superpixels_index:
        I_prime = update_instance_with_superpixels(I_prime, original_instance, index_of_superpixels)
        insertion_images.append(I_prime)

    new_predictions = ensemble_predict(models, insertion_images, x3)
    denorm_new_predictions = [new_pred * vott_target_test_std + vott_target_test_mean for new_pred in new_predictions]
    denormalized_H_new_predictions = [H_station - pred for pred in denorm_new_predictions]

    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions[1:]]
    initial_error = mean_squared_error(original_prediction, denormalized_H_new_predictions[0])

    print(f"Initial Prediction with Blurred Instance, new prediction: {denormalized_H_new_predictions[0]}, error: {initial_error}")
    only_inserted_pixel_new_predictions = denormalized_H_new_predictions[1:]

    for nr_superpixel, error in enumerate(errors):
        print(f"Inserted SuperPixel: {sorted_per_importance_all_superpixels_index[nr_superpixel]}, new prediction: {only_inserted_pixel_new_predictions[nr_superpixel]}, error: {error}")

    total_errors = [initial_error] + errors

    x = np.arange(0, len(total_errors))  # e.g., from 0 to 8
    x_for_auc = np.linspace(0, 1, len(total_errors))
    auc = calculate_auc(x_for_auc, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    return total_errors, auc

##### ***Deletion***

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def update_instance_removing_superpixels(current_instance, index_of_superpixels, std_zero_value=-0.6486319166678826):
    """
    Updates the instance by removing the selected superpixels.

    :param current_instance: Current instance.
    :param index_of_superpixels: List of indices for the current superpixel.
    :param std_zero_value: Value used to replace removed pixels (e.g., mean or zero).
    :return: Updated instance with the superpixel removed.
    """
    new_current_instance = current_instance.copy()

    for t in index_of_superpixels:
        new_current_instance[t, ..., 0] = std_zero_value
    return new_current_instance

def deletion(models, original_instance, x3_instance, sorted_per_importance_all_superpixels_index, original_prediction, H_station=390.0):
    """
    Computes the Deletion metric for a given explanation.

    :param models: List of pretrained models.
    :param original_instance: Original unperturbed instance.
    :param x3_instance: One-hot encoding for prediction.
    :param sorted_per_importance_all_superpixels_index: List of superpixel indices sorted by importance.
    :param original_prediction: Prediction on the original instance.
    :return: List of errors at each deletion step and the AUC.
    """
    deletion_images = []
    I_prime = copy.deepcopy(original_instance)

    for index_of_superpixels in sorted_per_importance_all_superpixels_index:
        I_prime = update_instance_removing_superpixels(I_prime, index_of_superpixels)
        deletion_images.append(I_prime)

    new_predictions = ensemble_predict(models, deletion_images, x3_instance)
    denorm_new_predictions = [new_pred * vott_target_test_std + vott_target_test_mean for new_pred in new_predictions]
    denormalized_H_new_predictions = [H_station - pred for pred in denorm_new_predictions]

    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions]

    initial_error = 0.0
    print(f"Initial Prediction with Original instance, prediction: {original_prediction}, error: {initial_error}")

    for nr_superpixel, error in enumerate(errors):
        print(f"Removed SuperPixel: {sorted_per_importance_all_superpixels_index[nr_superpixel]}, new prediction: {denormalized_H_new_predictions[nr_superpixel]}, error: {error}")

    total_errors = [initial_error] + errors

    x = np.arange(0, len(total_errors))  # e.g., from 0 to 8
    x_for_auc = np.linspace(0, 1, len(total_errors))
    auc = calculate_auc(x_for_auc, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    return total_errors, auc


####################################################################

def calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients, temporal_superpixels, models=vott_lstm_models_loaded, 
                                                           H_station=390.0, channel_prec=0, std_zero_value=-0.6486319166678826,input_size=(104,5,8,3),T=104,H=5,W=8):
  instance = copy.deepcopy(vottignasco_test_image[nr_instance])
  x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])

  abs_coefficients = np.abs(coefficients)

  input_size = T

  saliency_vector_i     = np.zeros(input_size)
  saliency_vector_i_abs = np.zeros(input_size)
  for i, (start,end,_) in enumerate(temporal_superpixels):
     saliency_vector_i[start:end+1]     = coefficients[i]
     saliency_vector_i_abs[start:end+1] = abs_coefficients[i] 

  # Compute ranking based on absolute values in the saliency vector
  all_superpixels_index = sorted_per_importance_superpixels_index_1D(saliency_vector_i_abs)

  # Insertion
  initial_blurred_instance = copy.deepcopy(instance)
  initial_blurred_instance[..., channel_prec] = std_zero_value
  original_prediction = ensemble_predict(models, instance, x3_instance)
  denormalized_H_original_prediction = H_station - (original_prediction * vott_target_test_std + vott_target_test_mean)

  errors_insertion, auc_insertion = insertion(models, instance, x3_instance, all_superpixels_index, initial_blurred_instance, denormalized_H_original_prediction)
  
  # Deletion
  errors_deletion, auc_deletion = deletion(models, instance, x3_instance, all_superpixels_index, denormalized_H_original_prediction)

  return saliency_vector_i, errors_insertion, auc_insertion, errors_deletion, auc_deletion


"""### ***Experiments***"""

from sklearn.linear_model import Ridge
import pickle
import numpy as np

# Channels
channel_prec, channel_tmax, channel_tmin = 0, 1, 2
channels = [channel_prec, channel_tmax, channel_tmin]

# Models and dimensions
models = vott_lstm_models_loaded
T, H, W, C = (104, 5, 8, 3)
input_size_temporal = T
std_zero_value = -0.6486319166678826

# Alpha and kernel width values
alpha_values = [0.1, 10.0]
percentile_kernel_width_values = [50, 90]

# Test set length
len_test_set = len(vottignasco_test_image)

# SHAP does not have hyperparameters in this case (temporal superpixels are the seasons)
# LIME is evaluated with combinations of p for kernel_width and alpha
# Dictionary to store the results
results = {"lime": {}, "shap": {}}

# Loop over test set instances
for nr_instance, _ in enumerate(vottignasco_test_image):
  print(f"######################## LIME-SHAP Temporal for instance #{nr_instance} ########################")
  base_start_time_lime_shap = datetime.datetime.now()
  
  superpixels, zs_primes, perturbed_instances, preds_masked = lime_shap(nr_instance, vottignasco_test_image, vottignasco_test_OHE, vottignasco_test_images_dates, channels, models,
                                                                        input_size_temporal)
  
  base_end_time_lime_shap = datetime.datetime.now()
  exec_time_base_lime_shap = base_end_time_lime_shap - base_start_time_lime_shap

  instance = copy.deepcopy(vottignasco_test_image[nr_instance])
  nr_coefficients = len(superpixels)

  # Prepare input for the regressor
  X = np.array([z.flatten() for z in zs_primes])  # Masks as rows
  y = np.array(preds_masked)  # Corresponding predictions

  ################ SHAP ##############################
  time_start_shap = datetime.datetime.now()

  # SHAP: compute weights and fit regression
  M = len(superpixels)
  weights_shap = calculate_weigths_shap(M, zs_primes)
  regressor_shap = Ridge(alpha=0.0)
  regressor_shap.fit(X, y, sample_weight=weights_shap)
  coefficients_shap = regressor_shap.coef_

  time_end_shap = datetime.datetime.now()
  exec_time_shap =  (exec_time_base_lime_shap + (time_end_shap - time_start_shap)).total_seconds()

  param_combination_shap = f"no_param_comb_shap"

  # Insertion/Deletion
  saliency_vector_i_shap, errors_insertion_shap,auc_insertion_shap, errors_deletion_shap,auc_deletion_shap =  calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients_shap, superpixels)

  # Initialize SHAP dictionary only once
  if param_combination_shap not in results["shap"]:
    results["shap"][param_combination_shap] = {
      "coefficients": [None] * len_test_set,  # List to support variable lengths
      "saliency_vectors": np.zeros((len_test_set, T)),
      "errors_insertion": len_test_set * [None],
      "auc_insertion": np.zeros((len_test_set, 1)),
      "errors_deletion": len_test_set * [None],
      "auc_deletion": np.zeros((len_test_set, 1)),
      "executions_times": np.zeros((len_test_set, 1)),
      "parameters_comb": param_combination_shap
    } 

  # Save coefficients for each instance
  results["shap"][param_combination_shap]["coefficients"][nr_instance]        = coefficients_shap
  results["shap"][param_combination_shap]["saliency_vectors"][nr_instance, :] = saliency_vector_i_shap
  results["shap"][param_combination_shap]["errors_insertion"][nr_instance]    = errors_insertion_shap
  results["shap"][param_combination_shap]["auc_insertion"][nr_instance, :]    = auc_insertion_shap
  results["shap"][param_combination_shap]["errors_deletion"][nr_instance]     = errors_deletion_shap
  results["shap"][param_combination_shap]["auc_deletion"][nr_instance, :]     = auc_deletion_shap
  results["shap"][param_combination_shap]["executions_times"][nr_instance, :] = exec_time_shap

  ################ END SHAP #####################################

  ################ LIME ######################################### 
  # LIME: loop over hyperparameters
  for alpha in alpha_values:
    for kernel_width_p in percentile_kernel_width_values:
      time_start_lime = datetime.datetime.now()

      weights_lime = calculate_weigths_lime(instance, perturbed_instances, percentile_kernel_width=kernel_width_p)
      regressor_lime = Ridge(alpha=alpha)
      regressor_lime.fit(X, y, sample_weight=weights_lime)
      coefficients_lime = regressor_lime.coef_

      time_end_lime = datetime.datetime.now()
      exec_time_lime = (exec_time_base_lime_shap + (time_end_lime - time_start_lime)).total_seconds()

      param_combination_lime = f"kw_{kernel_width_p}_alpha_{alpha}"

      # Ranking with Insertion/Deletion
      saliency_vector_i_lime, errors_insertion_lime,auc_insertion_lime, errors_deletion_lime,auc_deletion_lime = calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients_lime, superpixels)

      if param_combination_lime not in results["lime"]:
        results["lime"][param_combination_lime] = {
          "coefficients": [None] * len_test_set,
          "saliency_vectors": np.zeros((len_test_set, T)),
          "errors_insertion": len_test_set * [None],
          "auc_insertion": np.zeros((len_test_set, 1)),
          "errors_deletion": len_test_set * [None],
          "auc_deletion": np.zeros((len_test_set, 1)),
          "executions_times": np.zeros((len_test_set, 1)),
          "parameters_comb": param_combination_lime
        }

      results["lime"][param_combination_lime]["coefficients"][nr_instance]        = coefficients_lime
      results["lime"][param_combination_lime]["saliency_vectors"][nr_instance, :] = saliency_vector_i_lime
      results["lime"][param_combination_lime]["errors_insertion"][nr_instance]    = errors_insertion_lime
      results["lime"][param_combination_lime]["auc_insertion"][nr_instance, :]    = auc_insertion_lime
      results["lime"][param_combination_lime]["errors_deletion"][nr_instance]     = errors_deletion_lime
      results["lime"][param_combination_lime]["auc_deletion"][nr_instance, :]     = auc_deletion_lime
      results["lime"][param_combination_lime]["executions_times"][nr_instance, :] = exec_time_lime

# Save results once per setup
#path_to_save_results = f"/content/lime_shap_temporal_results.pkl"
#path_to_save_results = os.path.join(work_path, f"Water_Resources/rise-video/XAI/temporal/results/lime_shap_multiplicative_norm_zero/temporal_results.pkl")
path_to_save_results = f"{RESULT_DIR}/lime_shap_all_temporal_results.pkl"
with open(path_to_save_results, 'wb') as f:
  pickle.dump(results, f)


