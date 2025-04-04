
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
import sys
import datetime

"""
##### ***Data & Black-Box***

"""

RESULT_DIR = str(sys.argv[1])
print(f"RESULT_DIR: {RESULT_DIR}")

# IMPORT DATA FOR VOTTIGNASCO
import os

# Get the actual path from an environment variable
work_path = os.environ['WORK']  # Get the value of the WORK environment variable
v_test_OHE_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_month_OHE.npy"
v_test_image_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_normalized_image_sequences.npy"
v_test_target_dates_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_target_dates.npy"
v_test_normalization_factors_std_path = "###### REPLACE WITH REAL PATH #######/normalization_factors/Vottignasco_00425010001_training_target_std.npy"
v_test_normalization_factors_mean_path = "###### REPLACE WITH REAL PATH #######/normalization_factors/Vottignasco_00425010001_training_target_mean.npy"

# Load the numpy array from the files
vottignasco_test_OHE    = np.load(v_test_OHE_path)
vottignasco_test_image  = np.load(v_test_image_path)
vottignasco_test_dates  = np.load(v_test_target_dates_path)
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

### ***Temporal-RISE***

#### ***Generation Masks (1D): Uniform***

import numpy as np
from tqdm import tqdm

def generate_masks_1d(N, input_size, seed=42, **kwargs):
    """
    Parameters:
    - input_size: is the number of time-steps -> scalar
    """
    l = kwargs.get("l", 3)  # The length of the small_mask
    p1 = kwargs.get("p1", 0.5)  # Probability of mask activation

    np.random.seed(seed)

    # Generate a random 1D mask (length = small_mask_length)
    grid = np.random.rand(N, l) < p1
    grid = grid.astype('float32')  # Convert to float32 format

    # Create a structure for the final masks
    masks = np.empty((N, input_size))  # Final masks of size (N, H)

    for i in tqdm(range(N), desc='Generating masks'):
        # Calculate the interpolation points
        x = np.linspace(0, l - 1, l)  # Indices of the small mask
        new_x = np.linspace(0, l - 1, input_size)  # New points for the H dimension

        # 1D interpolation
        interpolated_mask = np.interp(new_x, x, grid[i])  # Interpolate the mask

        # Apply the interpolated mask to the final mask
        masks[i, :] = interpolated_mask

    # Filter out masks that are all 0.0
    #masks = masks[~(masks == 0).all(axis=1)]  # Filter along dimension 1 (H)

    return masks

#### ***Application Masks***

def multiplicative_uniform_noise_onechannel(images, masks, channel, **kwargs):
    std_zero_value = kwargs.get("std_zero_value", -0.6486319166678826)

    masked = []

    # Iterate over all the N generated masks
    for mask in masks:
        masked_images = copy.deepcopy(images)  # Deep copy of the original images

        for t in range(len(mask)):
          masked_images[t][..., channel] = masked_images[t][..., channel] * mask[t] + (1-mask[t]) * std_zero_value

        masked.append(masked_images)

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


"""#### ***Saliency Map***"""

def calculate_saliency_map_ev_masks(N, weights, masks):
    """
    Calculate the average saliency map given a series of predictions and masks.

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

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # Added the fraction 1/expected_value(masks)

    return np.squeeze(sal)

def calculate_saliency_map_ev_masks_cond_var(N, weights, ref, masks):
    """
    Calculate the average saliency map given a series of predictions and masks.
    CONDITIONAL VARIANCE

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

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # Added the fraction 1/expected_value(masks)

    return np.squeeze(sal)

def calculate_saliency_map_ev_masks_cond_bias(N, weights, ref, masks):
    """
    Calculate the average saliency map given a series of predictions and masks.
    CONDITIONAL BIAS

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

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # Added the fraction 1/expected_value(masks)

    return np.squeeze(sal)


"""#### ***Temporal-RISE: Framework***"""

def rise_temporal_explain(nr_instance, data_test_image, data_test_OHE, models, channel,
                          N, generate_masks_fn, seed, perturb_instance_fn, calculate_saliency_map_fn, H_station=390.0, **kwargs):
  print(f"############################### RISE-Temporal on Instance #{nr_instance} ###########################")
  instance    = copy.deepcopy(data_test_image[nr_instance])
  x3_instance = copy.deepcopy(data_test_OHE[nr_instance])

  input_size = instance.shape[0]
  
  masks = generate_masks_fn(N, input_size, seed, **kwargs)
  perturbed_instances = perturb_instance_fn(instance, masks, channel)

  # Prediction on Original Instance
  pred_original = ensemble_predict(models, instance, x3_instance)
  # Predictions on Perturbed Instances
  preds_masked = ensemble_predict(models, perturbed_instances, x3_instance)

  # Denormalize Black-Box Output with H_station 
  denorm_pred_original = (pred_original * vott_target_test_std) + vott_target_test_mean
  denorm_preds_masked  = [(pred_masked * vott_target_test_std) + vott_target_test_mean for pred_masked in preds_masked]
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
  print(f"############### Process completed. Saliency vector generated for Instance #{nr_instance} ###############")
  
  return np.squeeze(s1_i), np.squeeze(s2_i), np.squeeze(s3_i), np.squeeze(s4_i)


#### ***Evaluation Metric***

# I also implemented the batch approach here to improve execution times

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def calculate_auc(x, y):
    """
    Calculate the area under the curve (AUC) using the trapezoidal rule.

    :param x: x-axis values (fraction of pixels inserted).
    :param y: y-axis values (calculated errors).
    :return: Area under the curve.
    """
    return np.trapz(y, x)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def update_image_with_important_frame(current_instance, original_instance, t):
    """
    Update the current_instance by inserting the t-th frame.

    :param current_instance: current instance.
    :param original_instance: original instance.
    :param t: frame number to insert into the current_instance.
    :return: updated instance.
    """
    new_instance = copy.deepcopy(current_instance)
    new_instance[t,:,:,0] = original_instance[t,:,:,0]
    return new_instance


def insertion(models, original_images, x3, important_indices, initial_blurred_images, original_prediction, H_station=390.0):
    """
    Calculate the insertion metric for a given explanation.

    :param models: List of pre-trained models.
    :param original_images: Original image.
    :param x3: One-hot encoding for the prediction.
    :param important_indices: Indices of the pixels in order of importance.
    :param initial_blurred_images: Initial image with all pixels set to zero.
    :return: List of errors at each insertion step.
    """

    # Original prediction
    #original_prediction = ensemble_predict(models, original_images, x3)[0]
    #print("Original prediction:", original_prediction)

    # List to store instances where I gradually add frames
    insertion_images = [initial_blurred_images]

    # Prediction on the initial image (all pixels set to zero)
    I_prime = initial_blurred_images.copy()

    # Gradually add the most important frames. I get a list of all images with the frames added incrementally
    for t in important_indices:
        #print(frame)
        I_prime = update_image_with_important_frame(I_prime, original_images, t)
        insertion_images.append(I_prime)

    # Calculate predictions on the instances where frames have been gradually added
    new_predictions = ensemble_predict(models, insertion_images, x3)
    denorm_new_predictions  = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions  = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    # Calculate MSE for each prediction with respect to the original instance prediction (from the test set)
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions[1:]]

    initial_error = mean_squared_error(original_prediction, denormalized_H_new_predictions[0]) # mse for the image with all blurred frames
    print(f"Initial Prediction with ALL Blurred Frame, pred: {denormalized_H_new_predictions[0]}, error: {initial_error}")
    only_inserted_frame_new_predictions = denormalized_H_new_predictions[1:]
    for t, error in enumerate(errors):
      print(f"Frame {important_indices[t]}, new prediction: {only_inserted_frame_new_predictions[t]}, error: {error}")

    total_errors = [initial_error] + errors
    # Normalize the fraction of inserted pixels
    x = np.linspace(0, 1, len(total_errors))
    print("len total errors: ", len(total_errors))

    # Calculate AUC
    auc = calculate_auc(x, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    return total_errors, auc

def update_image_by_removing_frame(current_instance, t, std_zero_value=-0.6486319166678826):
    """
    Update the image by removing the most important pixels.

    :param current_instance: Current instance.
    :param original_instance: Original instance.
    :param t: frame number to remove.
    :return: Updated instance with the superpixel.
    """
    new_current_instance = current_instance.copy()
    new_current_instance[t,:,:,0] = std_zero_value
    return new_current_instance

def deletion(models, original_images, x3, important_indices, original_prediction, H_station=390.0):
    """
    Calculate the deletion metric for a given explanation.

    :param models: List of pre-trained models.
    :param original_images: Original image.
    :param x3: One-hot encoding for the prediction.
    :param important_indices: Indices of the pixels in order of importance.
    :return: List of errors at each deletion step.
    """

    # List to store images from which I gradually remove frames
    deletions_images = []

    # Initialization
    I_prime = original_images.copy()

    # Gradually remove the most important frames
    for t in important_indices:
        I_prime = update_image_by_removing_frame(I_prime, t)
        deletions_images.append(I_prime)

    # Calculate predictions on all the images where frames have been gradually removed
    new_predictions = ensemble_predict(models, deletions_images, x3)
    denorm_new_predictions  = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions  = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    # Calculate MSE with respect to the original prediction
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions]

    initial_error = 0.0
    print(f"Initial Prediction with Original Images, prediction: {original_prediction}, error: {initial_error}")
    for t, error in enumerate(errors):
      print(f"Removed frame {important_indices[t]}, new prediction: {denormalized_H_new_predictions[t]}, error: {error}")

    total_errors = [initial_error] + errors # Initial error + errors for all removed pixels

    # Normalize the fraction of removed pixels
    x = np.arange(len(total_errors)) / len(total_errors)

    # Calculate AUC
    auc = calculate_auc(x, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    return total_errors, auc


### Experiments

channel_prec = 0
models = vott_lstm_models_loaded
seed = 42
T,H,W,C = (104,5,8,3)
std_zero_value = -0.6486319166678826
H_station = 390.0

N = 1000

l_values = [16]
p_values = [0.5, 0.6, 0.7, 0.8, 0.9]

len_test_set = len(vottignasco_test_image)

for nr_setup,l in enumerate(l_values):
  print(f"############################## Setup #{nr_setup} ##############################")
  for _, p in enumerate(p_values):
    # Keep all sal_vecs for the entire Test-Set
    saliency_vectors = np.zeros((len_test_set,4,T))

    # Errors and AUC Insertion/Deletion for the entire Test-Set
    errors_insertion_all_testset = np.zeros((len_test_set,4, T+1))   # (105, 105)
    auc_insertion_all_testset    = np.zeros((len_test_set,4, 1))     # (105, 1)
    errors_deletion_all_testset  = np.zeros((len_test_set,4, T+1))   # (105, 105)
    auc_deletion_all_testset     = np.zeros((len_test_set,4, 1))     # (105, 1)

    nr_p = str(p).replace('.','') 
    param_combination = f"l_{l}_p_{nr_p}"

    execution_times = []

    print(f"############################## Parameters Combination: {param_combination} ##############################")
    for nr_instance,_ in enumerate(vottignasco_test_image):
       print(f"###################### Explanation for Instance #{nr_instance} ####################################")
       time_start = datetime.datetime.now()

       s1_i,s2_i,s3_i,s4_i = rise_temporal_explain(nr_instance, vottignasco_test_image, vottignasco_test_OHE, models, channel_prec,
                                                N, generate_masks_1d, seed, multiplicative_uniform_noise_onechannel, calculate_saliency_map_ev_masks, H_station, l=l, p1=p)

       time_end = datetime.datetime.now()
       exec_time = (time_end - time_start).total_seconds()
       execution_times.append(exec_time)

       # DEBUG
       print("s1 frame 0:", s1_i[0],"\n")
       print("s2 frame 0:", s2_i[0], "\n")
       print("s3 frame 0:", s3_i[0], "\n")
       print("s4 frame 0:", s4_i[0], "\n")

       saliency_vectors[nr_instance][0] = s1_i
       saliency_vectors[nr_instance][1] = s2_i
       saliency_vectors[nr_instance][2] = s3_i
       saliency_vectors[nr_instance][3] = s4_i

       instance    = copy.deepcopy(vottignasco_test_image[nr_instance])
       x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])

       # Insertion s2,s3,s4
       # Blurred video to start with for Insertion. All pixels of Prec set to std_zero_value
       initial_blurred_instance = copy.deepcopy(instance)
       initial_blurred_instance[:,:,:,channel_prec] = std_zero_value

       original_instance = copy.deepcopy(instance)
       original_prediction = ensemble_predict(models, original_instance, x3_instance)
       denormalized_H_original_prediction = H_station - (original_prediction * vott_target_test_std + vott_target_test_mean)
       print(f"Original Prediction: {denormalized_H_original_prediction}")

       all_important_frames_s2 = np.argsort(s2_i)[:] # Maximum importance for values close to 0
       all_important_frames_s3 = np.argsort(np.abs(s3_i))[:] # Maximum importance for values close to 0
       all_important_frames_s4 = np.argsort(s4_i)[:] # Maximum importance for values close to 0
    
       # Insertion
       errors_insertion_s2,auc_insertion_s2 = insertion(models, original_instance, x3_instance, all_important_frames_s2, initial_blurred_instance, denormalized_H_original_prediction) # s2   
       errors_insertion_s3,auc_insertion_s3 = insertion(models, original_instance, x3_instance, all_important_frames_s3, initial_blurred_instance, denormalized_H_original_prediction) # s3
       errors_insertion_s4,auc_insertion_s4 = insertion(models, original_instance, x3_instance, all_important_frames_s4, initial_blurred_instance, denormalized_H_original_prediction) # s4

       for nr_error in range (0, (T+1)):
         errors_insertion_all_testset[nr_instance][1][nr_error] = errors_insertion_s2[nr_error]
         errors_insertion_all_testset[nr_instance][2][nr_error] = errors_insertion_s3[nr_error]
         errors_insertion_all_testset[nr_instance][3][nr_error] = errors_insertion_s4[nr_error]

       auc_insertion_all_testset[nr_instance][1] = auc_insertion_s2
       auc_insertion_all_testset[nr_instance][2] = auc_insertion_s3
       auc_insertion_all_testset[nr_instance][3] = auc_insertion_s4

       # Deletion
       errors_deletion_s2,auc_deletion_s2 = deletion(models, original_instance, x3_instance, all_important_frames_s2, denormalized_H_original_prediction) # s2
       errors_deletion_s3,auc_deletion_s3 = deletion(models, original_instance, x3_instance, all_important_frames_s3, denormalized_H_original_prediction) # s3
       errors_deletion_s4,auc_deletion_s4 = deletion(models, original_instance, x3_instance, all_important_frames_s4, denormalized_H_original_prediction) # s4

       for nr_error in range (0, (T+1)):
         errors_deletion_all_testset[nr_instance][1][nr_error] = errors_deletion_s2[nr_error]
         errors_deletion_all_testset[nr_instance][2][nr_error] = errors_deletion_s3[nr_error]
         errors_deletion_all_testset[nr_instance][3][nr_error] = errors_deletion_s4[nr_error]

       auc_deletion_all_testset[nr_instance][1] = auc_deletion_s2
       auc_deletion_all_testset[nr_instance][2] = auc_deletion_s3
       auc_deletion_all_testset[nr_instance][3] = auc_deletion_s4

    print(f"#################################### END for all Instance in Test-Set for {param_combination} ####################################")
    
    result = {
                "saliency_vectors": saliency_vectors,
                "errors_insertion": errors_insertion_all_testset,
                "auc_insertion": auc_insertion_all_testset,
                "errors_deletion": errors_deletion_all_testset,
                "auc_deletion": auc_deletion_all_testset,
                "parameters_comb": param_combination,
                "execution_times": execution_times  # List of nr_instances (105) execution times for each instance
            }
    
    #path_to_save_results = os.path.join(work_path, f"Water_Resources/rise-video/XAI/temporal/results/new_rise_multiplicative_norm_zero/temporal_results_setup_{param_combination}_diff_pred.pkl")
    path_to_save_results = f"{RESULT_DIR}/rise_temporal_original_result_setup_{param_combination}.pkl"
    # Saving the results list in a pickle file
    with open(path_to_save_results, 'wb') as f:
      pickle.dump(result, f)

print("############################# END FOR ALL SETUPS ##########################################################################")
