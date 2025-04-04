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
# Save Execution Time
import datetime
import sys

"""
##### ***Data & Black-Box***

"""
RESULT_DIR = str(sys.argv[1])
print(f"RESULT_DIR: {RESULT_DIR}")

# IMPORTING DATA FOR VOTTIGNASCO
import os

# Get the actual path from an environment variable
#work_path = os.environ['WORK']  # Get the value of the WORK environment variable
v_test_OHE_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_month_OHE.npy"
v_test_image_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_normalized_image_sequences.npy"
v_test_target_dates_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_target_dates.npy"
v_test_normalization_factors_std_path = "###### REPLACE WITH REAL PATH #######/normalization_factors/Vottignasco_00425010001_training_target_std.npy"
v_test_normalization_factors_mean_path = "###### REPLACE WITH REAL PATH #######/normalization_factors/Vottignasco_00425010001_training_target_mean.npy"

# Load the numpy arrays from the files
vottignasco_test_OHE    = np.load(v_test_OHE_path)
vottignasco_test_image  = np.load(v_test_image_path)
vottignasco_test_dates  = np.load(v_test_target_dates_path)
vott_target_test_std    = np.load(v_test_normalization_factors_std_path) 
vott_target_test_mean   = np.load(v_test_normalization_factors_mean_path)

# """##### ***Black Boxes***""

# If you want to enable dropout at runtime
mc_dropout = True

# Definition of the custom class doprout_custom
class doprout_custom(tf.keras.layers.SpatialDropout1D):
    def call(self, inputs, training=None):
        if mc_dropout:
            return super().call(inputs, training=True)
        else:
            return super().call(inputs, training=False)

# Path to the directory on Cineca
base_dir  = "###### REPLACE WITH REAL PATH #######/trained_models/seq2val/Vottignasco"
lstm_suffix = 'time_dist_LSTM'

vott_lstm_models = []

def extract_index(filename):
    """Function to extract the final index from the file name."""
    return int(filename.split('_LSTM_')[-1].split('.')[0])

# Find all the .keras files in the folder and add them to the list
for filename in os.listdir(base_dir):
    if lstm_suffix in filename and filename.endswith(".keras"):
        vott_lstm_models.append(os.path.join(base_dir, filename))

# Sort the models based on the final index
vott_lstm_models = sorted(vott_lstm_models, key=lambda x: extract_index(os.path.basename(x)))

# List for the loaded models
vott_lstm_models_loaded = []

for i, model_lstm_path in enumerate(vott_lstm_models[:10]):  # Taking the first 10 sorted models
    #print(f"Loading LSTM model {i+1}: {model_lstm_path}")

    # Load the model with the custom class
    model = load_model(model_lstm_path, custom_objects={"doprout_custom": doprout_custom})

    # Add the model to the list
    vott_lstm_models_loaded.append(model)

print(vott_lstm_models_loaded)


"""### ***RISE-Spatio_Temporal***

#### ***Generation Masks (3D): Uniform***
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm


def generate_masks_3d(N, input_size, seed=42, **kwargs):
    """
    Parameters:
    - input_size: (t, h, w) -> final size of the 3D mask (time, Height, Width)
    """
    T, H, W = input_size
    l = kwargs.get("l", 8)   # Size of the small mask for time
    h = kwargs.get("h", 2)
    w = kwargs.get("w", 4)
    p1 = kwargs.get("p1", 0.5)  # Activation probability

    np.random.seed(seed)

    # Generate a random 3D mask of size (l, h, w)
    grid = np.random.rand(N, l, h, w) < p1
    grid = grid.astype('float32')

    # Structure for the final masks of size (N, T, H, W)
    masks = np.empty((N, T, H, W))

    # Coordinates for spatial interpolation
    grid_x = np.linspace(0, h - 1, h)
    grid_y = np.linspace(0, w - 1, w)
    grid_t = np.linspace(0, l - 1, l)

    for i in tqdm(range(N), desc='Generating masks'):
        # Create an interpolator for the current mask
        interpolator = RegularGridInterpolator(
            (grid_t, grid_x, grid_y), grid[i], method='linear', bounds_error=False, fill_value=0
        )
        new_t = np.linspace(0, l - 1, T)
        new_x = np.linspace(0, h - 1, H)
        new_y = np.linspace(0, w - 1, W)
        mesh_t, mesh_x, mesh_y = np.meshgrid(new_t, new_x, new_y, indexing='ij')
        points = np.stack((mesh_t, mesh_x, mesh_y), axis=-1).reshape(-1, 3)
        interpolated_mask = interpolator(points)
        masks[i] = interpolated_mask.reshape(T, H, W)

    return masks

"""#### ***Application Masks***"""

def multiplicative_uniform_noise_onechannel(images, masks, channel, **kwargs):
    """
    Applies continuous multiplicative noise in 3D to the images based on the provided masks.

    Parameters:
    - images: array of shape (104, 5, 8, C) -> time series of images with multiple channels
    - masks: array of shape (N, 104, 5, 8) -> continuous masks with values [0, 1]
    - channel: specific channel to apply noise on (0: Prec, 1: TMax, 2: TMin)
    - std_zero_value: value of the disturbance to apply where mask = 0
    """
    std_zero_value = kwargs.get("std_zero_value", -0.6486319166678826)

    masked = []

    # Iterate over all the generated N masks
    for mask in masks:
        masked_images = copy.deepcopy(images)  # Deep copy of the original images

        # Extract only the desired channel
        channel_values = masked_images[..., channel]

        # Apply the formula: v(p) + z (1-p)
        perturbed_values = channel_values * mask + (1 - mask) * std_zero_value

        # Overwrite the channel with the perturbed values
        masked_images[..., channel] = perturbed_values

        masked.append(masked_images)

    return masked

def ensemble_predict(models, images, x3_exp, batch_size=1000):
    # Ensure images is a list
    if not isinstance(images, list):
        images = [images]

    len_x3 = len(images)

    # Convert x3_exp into a tensor replicated for each image
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

        # Convert batch predictions to tensor and compute the mean
        batch_preds_tensor = tf.stack(batch_preds)
        mean_batch_preds = tf.reduce_mean(batch_preds_tensor, axis=0)

        # Add the batch predictions to the final list
        final_preds.extend(mean_batch_preds.numpy())

    return np.array(final_preds)


"""#### ***Saliency Map***"""

def calculate_saliency_map_ev_masks(N, weights, masks):
    """
    Calculates the average saliency map given a series of predictions and masks.

    :param weights_array: Array of predictions (mask weights).
    :param masks: Array of masks (number of masks x mask dimensions).
    :return: Average saliency map.
    """
    sal = []
    
    for j in range(len(masks)):
        sal_j = weights[j] * masks[j]
        sal.append(sal_j)

    # Now calculate the mean along axis 0
    ev_masks = np.mean(masks, axis=0)

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # adding the fraction 1/expected_value(masks)

    return np.squeeze(sal)

def calculate_saliency_map_ev_masks_cond_var(N, weights, ref, masks):
    """
    Calculates the average saliency map given a series of predictions and masks.
    CONDITIONED VARIANCE

    :param weights_array: Array of predictions (mask weights).
    :param masks: Array of masks (number of masks x mask dimensions).
    :return: Average saliency map.
    """
    sal = []
    
    for j in range(len(masks)):
        sal_j = ((weights[j] - ref)**2) * masks[j]
        sal.append(sal_j)

    # Now calculate the mean along axis 0
    ev_masks = np.mean(masks, axis=0)

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # adding the fraction 1/expected_value(masks)

    return np.squeeze(sal)

def calculate_saliency_map_ev_masks_cond_bias(N, weights, ref, masks):
    """
    Calculates the average saliency map given a series of predictions and masks.
    CONDITIONED VARIANCE

    :param weights_array: Array of predictions (mask weights).
    :param masks: Array of masks (number of masks x mask dimensions).
    :return: Average saliency map.
    """
    sal = []
    
    for j in range(len(masks)):
        sal_j = (weights[j] - ref) * masks[j]
        sal.append(sal_j)

    # Now calculate the mean along axis 0
    ev_masks = np.mean(masks, axis=0)

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # adding the fraction 1/expected_value(masks)

    return np.squeeze(sal)

"""#### ***Spatio_Temporal-RISE: Framework***"""

def rise_st_explain(nr_instance, data_test_image, data_test_OHE, models, channel,
                    N, generate_masks_fn, seed, perturb_instance_fn, calculate_saliency_map_fn, H_station=390.0, **kwargs):
  print(f"############################### RISE-Temporal on Instance #{nr_instance} ###########################")
  instance    = copy.deepcopy(data_test_image[nr_instance])
  x3_instance = copy.deepcopy(data_test_OHE[nr_instance])

  input_size = (instance.shape[0], instance.shape[1], instance.shape[2])

  masks = generate_masks_fn(N, input_size, seed, **kwargs)
  perturbed_instances = perturb_instance_fn(instance, masks, channel)

  # Prediction on Original Instance
  pred_original = ensemble_predict(models, instance, x3_instance)
  # Predictions on Perturbed Instances
  preds_masked = ensemble_predict(models, perturbed_instances, x3_instance)
  # Denormalization with H_station (station height)
  denorm_pred_original = pred_original * vott_target_test_std + vott_target_test_mean
  denorm_preds_masked  = [pred_masked * vott_target_test_std + vott_target_test_mean for pred_masked in preds_masked]
  denormalized_H_pred_original = H_station - denorm_pred_original
  denormalized_H_preds_masked  = [H_station - denorm_pred_masked for denorm_pred_masked in denorm_preds_masked]

  # Mask Weights
  weights = np.concatenate(denormalized_H_preds_masked, axis=0)

  ### S1 
  s1_i = calculate_saliency_map_fn(N, weights, masks)
  ### S2
  s2_i = calculate_saliency_map_ev_masks_cond_var(N, weights, s1_i, masks)
  ### S3 (BIAS)
  s3_i = calculate_saliency_map_ev_masks_cond_bias(N, weights, denormalized_H_pred_original, masks)
  ### S4 (RMSE)
  s4_i = np.sqrt(calculate_saliency_map_ev_masks_cond_var(N, weights, denormalized_H_pred_original, masks))
  print(f"############### Process completed. Saliency map generated for Instance #{nr_instance} ###############")

  return np.squeeze(s1_i), np.squeeze(s2_i), np.squeeze(s3_i), np.squeeze(s4_i)


"""#### ***Evaluation Metrics***"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def calculate_auc(x, y):
    """
    Calculates the area under the curve (AUC) using the trapezoidal method.

    :param x: x-axis values (fraction of pixels/frames inserted).
    :param y: y-axis values (calculated errors).
    :return: Area under the curve.
    """
    return np.trapz(y, x)

import numpy as np

def get_flatten_saliency_video_ordered_by_importance(saliency_video):
    """
    Receives a 3D array (T, H, W) with saliency values and returns
    a list of pixels sorted by importance.

    :param saliency_video: np.array of shape (T, H, W) with saliency values.
    :return: Sorted list of tuples (saliency, [t, x, y]), with decreasing saliency.
    """
    # Flatten the saliency values
    flatten_saliency_video = saliency_video.flatten()

    # Get the original indices [t, x, y] for each pixel
    indices = np.array(np.unravel_index(np.arange(flatten_saliency_video.size), saliency_video.shape)).T

    # Combine values and indices: list of tuples (saliency, [t, x, y])
    saliency_video_value_with_indices = list(zip(flatten_saliency_video, indices))

    # Sort by saliency value (in descending order) and, in case of ties, by coordinates (in descending order)
    sorted_saliency = sorted(
        saliency_video_value_with_indices,
        key=lambda x: (x[0], -x[1][0], -x[1][1], -x[1][2]),
        reverse=True
    )

    return sorted_saliency


def plot_insertion_curve(total_errors, auc, title="Insertion Metric Curve"):
    """
    Plots the insertion metric curve with mean squared error.

    :param total_errors: List of error values for each fraction of pixels inserted.
    :param auc: Calculated Area Under Curve (AUC) value.
    :param title: Plot title (default: "Insertion Metric Curve").
    """

    # New X axis normalized between 0 and 1
    x = np.linspace(0, 1, len(total_errors))

    # Plot the error curve and the area under the curve (AUC)
    plt.figure(figsize=(7, 5))
    plt.plot(x, total_errors, linestyle='-', color='blue')

    # Area under the curve
    plt.fill_between(x, total_errors, color='skyblue', alpha=0.4)

    # "Error curve" text in the top-right with smaller font
    plt.legend(['Error curve'], loc='upper right', fontsize=9)

    # AUC text just below "Error curve"
    plt.text(x[-1] - 0.02, max(total_errors) * 0.9,
             f'AUC: {auc:.2f}',
             horizontalalignment='right',
             fontsize=8,
             bbox=dict(facecolor='white', alpha=0.5))

    # Axis labels
    plt.xlabel('Fraction of pixels inserted')
    plt.ylabel('Mean Squared Error')

    # Plot title
    plt.title(title)

    # Show the plot
    plt.show()


def plot_deletion_curve(total_errors, auc, title="Deletion Metric Curve"):
    """
    Plots the deletion metric curve with mean squared error.

    :param total_errors: List of error values for each fraction of pixels removed.
    :param auc: Calculated Area Under Curve (AUC) value.
    :param title: Plot title (default: "Deletion Metric Curve").
    """

    # Normalize the X axis between 0 and 1
    x = np.linspace(0, 1, len(total_errors))

    # Create the plot
    plt.figure(figsize=(7,5))
    plt.plot(x, total_errors, linestyle='-', color='red')

    # Area under the curve
    plt.fill_between(x, total_errors, color='lightcoral', alpha=0.4)

    # "Error curve" text in the top-left with smaller font
    plt.legend(['Error curve'], loc='lower right',  bbox_to_anchor=(0.97, 0.02))

    # AUC text slightly below the legend
    plt.text(0.941, 0.13, f'AUC: {auc:.2f}',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='grey'))

    # Axis labels
    plt.xlabel('Fraction of pixels removed')
    plt.ylabel('Mean Squared Error')

    # Plot title
    plt.title(title)

    # Show the plot
    plt.show()

"""##### ***Insertion***"""

def update_instance_with_pixels(current_instance, original_instance, t, x, y):
    """
    Updates the image by inserting the most important pixels.

    :param current_instance: Current instance.
    :param original_instance: Original instance.
    :param t: t-coordinate of the pixel to insert
    :param x: x-coordinate of the pixel to insert
    :param y: y-coordinate of the pixel to insert
    :return: Updated instance with the inserted pixel.
    """
    new_current_instance = current_instance.copy()
    new_current_instance[t, x, y, 0] = original_instance[t, x, y, 0]

    return new_current_instance


def insertion(model, original_instance, x3_instance, sorted_per_importance_pixels_index, initial_blurred_instance, original_prediction, H_station=390.0):
    """
    Calculates the insertion metric for a given explanation.

    :param model: Black-box model.
    :param original_instance: Original instance.
    :param sorted_per_importance_pixels_index: List of lists of all superpixels by importance.
    :param initial_blurred_images: Initial image with all pixels set to zero.
    :return: List of errors at each insertion step.
    """

    # List to store instances with progressively added pixels. Initialized with the initial blurred instance
    insertion_images = [initial_blurred_instance]

    # Prediction on the initial image (all pixels set to zero)
    I_prime = copy.deepcopy(initial_blurred_instance)

    # Gradually add the most important pixels (per frame). This creates a list with all images with pixels gradually added
    for t, x, y in sorted_per_importance_pixels_index:
        I_prime = update_instance_with_pixels(I_prime, original_instance, t, x, y)
        insertion_images.append(I_prime)

    insertion_images = [img.astype(np.float32) for img in insertion_images]
    # Calculate predictions on instances with progressively added pixels
    new_predictions = ensemble_predict(model, insertion_images, x3_instance)
    denorm_new_predictions  = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions  = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    # For each prediction, calculate the MSE with respect to the prediction on the original instance (as per test set). Ignore the first one, which is on the blurred image.
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions[1:]]

    initial_error = mean_squared_error(original_prediction, denormalized_H_new_predictions[0])
    print(f"Initial Prediction with Blurred Instance. Prediction: {denormalized_H_new_predictions[0]}, error: {initial_error}")
    
    #only_inserted_pixel_new_predictions = denormalized_H_new_predictions[1:]
    #for nr_pixel, error in enumerate(errors):
    #  print(f"Inserted Pixel: {sorted_per_importance_pixels_index[nr_pixel]}. Prediction: {only_inserted_pixel_new_predictions[nr_pixel]}, error: {error}")

    total_errors = [initial_error] + errors # Initial error + errors for all inserted pixels

    # New X axis
    x = np.linspace(0, 1, len(total_errors))
    # Calculate AUC with the new X axis
    auc = calculate_auc(x, total_errors)
    print(f"Area under the curve (AUC): {auc}")
    return total_errors, auc

"""##### ***Deletion***"""

def update_image_by_removing_pixels(current_instance, t, x, y, std_zero_value=-0.6486319166678826):
    """
    Updates the image by removing the specified x, y pixels.

    :param current_instance: Current instance.
    :param t: t-coordinate of the pixel to remove
    :param x: x-coordinate of the pixel to remove
    :param y: y-coordinate of the pixel to remove
    :return: Updated instance with t,x,y
    """
    new_instance = copy.deepcopy(current_instance)
    new_instance[t, x, y, 0] = std_zero_value # Set pixels to zero normalized for Prec
    # new_instance[t, x, y, 1] = 0.0 # Set pixels to zero for Tmax
    # new_instance[t, x, y, 2] = 0.0 # Set pixels to zero for Tmin
    return new_instance


def deletion(models, original_instance, x3_instance, sorted_per_importance_pixels_index, original_prediction, H_station=390.0):
    """
    Calculates the deletion metric for a given explanation.

    :param models: List of pre-trained models.
    :param original_instance: Original image.
    :param x3_instance: One-hot encoding for prediction.
    :param sorted_per_importance_pixels_index: Indices of pixels in order of importance.
    :return: List of errors, AUC at each deletion step.
    """
    # List to store images with pixels gradually removed (per time-step)
    deletion_images = []

    # Initialization
    I_prime = copy.deepcopy(original_instance)

    # Gradually remove the most important pixels (per frame). This creates a list with all images with pixels removed
    for t, x, y in sorted_per_importance_pixels_index:
        I_prime = update_image_by_removing_pixels(I_prime, t, x, y)
        deletion_images.append(I_prime)

    # Calculate prediction on all images with progressively removed pixels
    new_predictions = ensemble_predict(models, deletion_images, x3_instance)
    denorm_new_predictions  = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions  = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    # Calculate MSE with respect to the original prediction
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions]

    initial_error = 0.0
    print(f"Initial Prediction with Original Images, prediction: {original_prediction}, error: {initial_error}")

    #for nr_pixel, error in enumerate(errors):
    #    print(f"Removed pixel {sorted_per_importance_pixels_index[nr_pixel]}, new prediction: {denormalized_H_new_predictions[nr_pixel]}, error: {error}")

    total_errors = [initial_error] + errors  # Initial error + errors for all removed pixels

    # Normalize the fraction of pixels removed
    x = np.linspace(0, 1, len(total_errors))
    # Calculate AUC
    auc = calculate_auc(x, total_errors)

    print(f"Area under the curve (AUC): {auc}")
    return total_errors, auc


"""### ***Experiments***"""

channel_prec = 0
models = vott_lstm_models_loaded
seed = 42
T,H,W,C = (104,5,8,3)
std_zero_value = -0.6486319166678826
H_station = 390.0

N = 2500

l_values = [12]
h_w_values = [(2,4)]
p_values = [0.5, 0.6, 0.7, 0.8, 0.9]

len_test_set = len(vottignasco_test_image)

for nr_setup,l in enumerate(l_values):
  print(f"############################## Setup #{nr_setup} ##############################")
  for h,w in h_w_values:
    for _, p in enumerate(p_values):
      # Keep all saliency_videos for the entire test set
      saliency_videos = np.zeros((len_test_set,4, T, H, W))
       # Errors and AUC Insertion/Deletion for the entire test set
      errors_insertion_all_testset = np.zeros((len_test_set,4, (T*H*W)+1))   # (105, 4160+1)
      auc_insertion_all_testset    = np.zeros((len_test_set,4, 1))           # (105, 1)
      errors_deletion_all_testset  = np.zeros((len_test_set,4, (T*H*W)+1))   # (105, 4160+1)
      auc_deletion_all_testset     = np.zeros((len_test_set,4, 1))           # (105, 1)

      nr_p = str(p).replace('.','')
      
      param_combination = f"l_{l}_h{h}_w{w}_p_{nr_p}"

      print(f"############################## Parameters Combination: {param_combination} ##############################")
      execution_times = []
      for nr_instance,_ in enumerate(vottignasco_test_image):
        print(f"###################### Explanation for Instance #{nr_instance} ####################################")
        time_start = datetime.datetime.now()

        s1_i,s2_i,s3_i,s4_i = rise_st_explain(nr_instance, vottignasco_test_image, vottignasco_test_OHE, models, channel_prec,
                                              N, generate_masks_3d, seed, multiplicative_uniform_noise_onechannel, calculate_saliency_map_ev_masks, l=l, h=h, w=w, p1=p)

        time_end = datetime.datetime.now()
        exec_time = (time_end - time_start).total_seconds()

        execution_times.append(exec_time)

        saliency_videos[nr_instance][0] = s1_i
        saliency_videos[nr_instance][1] = s2_i
        saliency_videos[nr_instance][2] = s3_i
        saliency_videos[nr_instance][3] = s4_i

        instance    = copy.deepcopy(vottignasco_test_image[nr_instance])
        x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])

        # Insertion s2,s3,s4
        # Blurred video to start from for Insertion. All Prec pixels set to std_zero_value
        initial_blurred_instance = copy.deepcopy(instance)
        initial_blurred_instance[:,:,:,channel_prec] = std_zero_value

        original_instance = copy.deepcopy(instance)
        original_prediction = ensemble_predict(models, original_instance, x3_instance)
        denormalized_H_original_prediction = H_station - (original_prediction * vott_target_test_std + vott_target_test_mean)
        print(f"Original Prediction: {denormalized_H_original_prediction}")

        all_important_pixels_s2 = get_flatten_saliency_video_ordered_by_importance(s2_i)[::-1] # Values closest to 0 are "most important"
        all_important_pixels_s2 = [coord for _,coord in all_important_pixels_s2]

        all_important_pixels_s3 = get_flatten_saliency_video_ordered_by_importance(np.abs(s3_i))[::-1]
        all_important_pixels_s3 = [coord for _,coord in all_important_pixels_s3]

        all_important_pixels_s4 = get_flatten_saliency_video_ordered_by_importance(s4_i)[::-1]
        all_important_pixels_s4 = [coord for _,coord in all_important_pixels_s4]

        # Insertion
        errors_insertion_s2,auc_insertion_s2 = insertion(models, original_instance, x3_instance, all_important_pixels_s2, initial_blurred_instance, denormalized_H_original_prediction) # s2   
        errors_insertion_s3,auc_insertion_s3 = insertion(models, original_instance, x3_instance, all_important_pixels_s3, initial_blurred_instance, denormalized_H_original_prediction) # s3
        errors_insertion_s4,auc_insertion_s4 = insertion(models, original_instance, x3_instance, all_important_pixels_s4, initial_blurred_instance, denormalized_H_original_prediction) # s4

        for nr_error in range (0, (T*H*W)+1):
          errors_insertion_all_testset[nr_instance][1][nr_error] = errors_insertion_s2[nr_error]
          errors_insertion_all_testset[nr_instance][2][nr_error] = errors_insertion_s3[nr_error]
          errors_insertion_all_testset[nr_instance][3][nr_error] = errors_insertion_s4[nr_error]

        auc_insertion_all_testset[nr_instance][1] = auc_insertion_s2
        auc_insertion_all_testset[nr_instance][2] = auc_insertion_s3
        auc_insertion_all_testset[nr_instance][3] = auc_insertion_s4

        # Deletion
        errors_deletion_s2,auc_deletion_s2 = deletion(models, original_instance, x3_instance, all_important_pixels_s2, denormalized_H_original_prediction) # s2
        errors_deletion_s3,auc_deletion_s3 = deletion(models, original_instance, x3_instance, all_important_pixels_s3, denormalized_H_original_prediction) # s3
        errors_deletion_s4,auc_deletion_s4 = deletion(models, original_instance, x3_instance, all_important_pixels_s4, denormalized_H_original_prediction) # s4

        for nr_error in range (0, (T*H*W)+1):
          errors_deletion_all_testset[nr_instance][1][nr_error] = errors_deletion_s2[nr_error]
          errors_deletion_all_testset[nr_instance][2][nr_error] = errors_deletion_s3[nr_error]
          errors_deletion_all_testset[nr_instance][3][nr_error] = errors_deletion_s4[nr_error]

        auc_deletion_all_testset[nr_instance][1] = auc_deletion_s2
        auc_deletion_all_testset[nr_instance][2] = auc_deletion_s3
        auc_deletion_all_testset[nr_instance][3] = auc_deletion_s4

      print(f"#################################### END for all Instance in Test-Set for {param_combination} ####################################")

      result = {
        "saliency_videos": saliency_videos,
        "errors_insertion": errors_insertion_all_testset,
        "auc_insertion": auc_insertion_all_testset,
        "errors_deletion": errors_deletion_all_testset,
        "auc_deletion": auc_deletion_all_testset,
        "parameters_comb": param_combination,
        "execution_times": execution_times  # List of execution times for each instance
      }

      #path_to_save_results = os.path.join(work_path, f"Water_Resources/rise-video/XAI/spatial_temporal/results//test_st_results_setup_{param_combination}.pkl")
      path_to_save_results = f"{RESULT_DIR}/rise_st_original_result_setup_{param_combination}.pkl"
      # Saving the result list in a pickle file
      with open(path_to_save_results, 'wb') as f:
        pickle.dump(result, f)

print("############################# END FOR ALL SETUPS ##########################################################################")
