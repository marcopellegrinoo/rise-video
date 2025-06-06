# -*- coding: utf-8 -*-
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
# Save Execution Time
import datetime

"""

##### ***Data & Black-Box***

"""

RESULT_DIR = str(sys.argv[1])
print(f"RESULT_DIR: {RESULT_DIR}")

# Get the actual path from an environment variable
v_test_OHE_path = "C/Vottignasco_00425010001_test_month_OHE.npy"
v_test_image_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_normalized_image_sequences.npy"
v_test_target_dates_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_test_target_dates.npy"
v_test_normalization_factors_std_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_training_target_std.npy"
v_test_normalization_factors_mean_path = "###### REPLACE WITH REAL PATH #######/Vottignasco_00425010001_training_target_mean.npy"

# Load the numpy array from the files
vottignasco_test_OHE    = np.load(v_test_OHE_path)
vottignasco_test_image  = np.load(v_test_image_path)
vottignasco_test_dates  = np.load(v_test_target_dates_path)
vott_target_test_std    = np.load(v_test_normalization_factors_std_path) 
vott_target_test_mean   = np.load(v_test_normalization_factors_mean_path)

print(len(vottignasco_test_dates))
print(len(vottignasco_test_image))
print(len(vottignasco_test_OHE))

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
# base_dir = os.path.join(os.environ['WORK'], "Water_Resources/rise-video/trained_models/seq2val/Vottignasco")
base_dir  = "###### REPLACE WITH REAL PATH #######/Vottignasco"
lstm_suffix = 'time_dist_LSTM'

vott_lstm_models = []

def extract_index(filename):
    """Function to extract the final index from the filename."""
    return int(filename.split('_LSTM_')[-1].split('.')[0])

# Find all .keras files in the folder and add them to the list
for filename in os.listdir(base_dir):
    if lstm_suffix in filename and filename.endswith(".keras"):
        vott_lstm_models.append(os.path.join(base_dir, filename))

# Sort the models based on the final index
vott_lstm_models = sorted(vott_lstm_models, key=lambda x: extract_index(os.path.basename(x)))

# List for loaded models
vott_lstm_models_loaded = []

for i, model_lstm_path in enumerate(vott_lstm_models[:10]):  # Take the first 10 sorted models
    #print(f"Loading LSTM model {i+1}: {model_lstm_path}")

    # Load the model with the custom class
    model = load_model(model_lstm_path, custom_objects={"doprout_custom": doprout_custom})

    # Add the model to the list
    vott_lstm_models_loaded.append(model)

print(vott_lstm_models_loaded)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_rise_masks_2d(N, input_size, seed, **kwargs):
    """
    Generates N RISE masks for an image of size HxW.

    Parameters:
    - N: number of masks
    - input_size: (H, W) dimensions of the original image
    - h, w: dimensions of the low-resolution masks
    - p: probability of activating pixels in the initial binary mask

    Returns:
    - masks: array of shape (N, H, W) containing the normalized masks.
    """

    h  = kwargs.get("h", 2)
    w = kwargs.get("w", 2)
    p = kwargs.get("p", 0.5)

    np.random.seed(seed)

    masks = []
    H, W = input_size
    CH, CW = H // h, W // w  # Upscaling factor

    for _ in range(N):
        # 1. Generate the initial binary mask (h x w)
        small_mask = np.random.rand(h, w) < p

        up_size_h = (h+1) * CH
        up_size_w = (w+1) * CW

        # 2. Bilinear upsampling to the size (H + CH, W + CW)
        upsampled_mask = cv2.resize(small_mask.astype(np.float32),
                                    (up_size_w, up_size_h), interpolation=cv2.INTER_LINEAR)
        
        # 3. Random crop of the H x W region
        x_offset = np.random.randint(0, (up_size_h - H) + 1)
        y_offset = np.random.randint(0, (up_size_w - W) + 1)
        final_mask = upsampled_mask[x_offset:x_offset + H, y_offset:y_offset + W] 

        masks.append(final_mask)

    return np.array(masks)



def multiplicative_uniform_noise_onechannel(images, masks, channel, **kwargs):
    std_zero_value = kwargs.get("std_zero_value", -0.6486319166678826)

    masked = []

    # Loop over all the N generated masks
    for mask in masks:
        masked_images = copy.deepcopy(images)  # Deep copy of the original images

        # Apply the perturbation only to the specified channel
        masked_images[..., channel] = (
            masked_images[..., channel] * mask + (1 - mask) * std_zero_value)

        masked.append(masked_images)

    return masked

def ensemble_predict(models, images, x3_exp, batch_size=1000):
    # Make sure that images is a list
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

        # Convert the batch predictions to a tensor and calculate the mean
        batch_preds_tensor = tf.stack(batch_preds)
        mean_batch_preds = tf.reduce_mean(batch_preds_tensor, axis=0)

        # Add the batch predictions to the final list
        final_preds.extend(mean_batch_preds.numpy())

    return np.array(final_preds)


#### ***Saliency Map***

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

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # Added fraction 1/expected_value(masks)

    return np.squeeze(sal)

def calculate_saliency_map_ev_masks_cond_var(N, weights, ref, masks):
    """
    Calculates the average saliency map given a series of predictions and masks.
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

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # Added fraction 1/expected_value(masks)

    return np.squeeze(sal)

def calculate_saliency_map_ev_masks_cond_bias(N, weights, ref, masks):
    """
    Calculates the average saliency map given a series of predictions and masks.
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

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # Added fraction 1/expected_value(masks)

    return np.squeeze(sal)


#### ***Spatial-RISE: Framework***

def rise_spatial_explain(nr_instance, data_test_image, data_test_OHE, models, channel,
                          N, generate_masks_fn, seed, perturb_instance_fn, calculate_saliency_map_fn, H_station=390.0, **kwargs):
    print(f"############################### RISE-Spatial on Instance #{nr_instance} ###########################")
    instance    = copy.deepcopy(data_test_image[nr_instance])  # Copy the test image for the instance
    x3_instance = copy.deepcopy(data_test_OHE[nr_instance])   # Copy the OHE data for the instance

    input_size = (instance.shape[1], instance.shape[2])  # Get the dimensions of the image

    masks = generate_masks_fn(N, input_size, seed, **kwargs)  # Generate the masks using the provided function
    perturbed_instances = perturb_instance_fn(instance, masks, channel)  # Perturb the instances with the generated masks

    # Prediction on the original instance
    pred_original = ensemble_predict(models, instance, x3_instance)  # Get the prediction for the original instance
    # Predictions on the perturbed instances
    preds_masked = ensemble_predict(models, perturbed_instances, x3_instance)  # Get predictions for the perturbed instances

    # Denormalize the output from the Black-Box model using H_station
    denorm_pred_original = (pred_original * vott_target_test_std) + vott_target_test_mean
    denorm_preds_masked  = [(pred_masked * vott_target_test_std) + vott_target_test_mean for pred_masked in preds_masked]
    
    # Denormalized predictions with H_station (i.e., target station)
    denormalized_H_pred_original = H_station - denorm_pred_original
    denormalized_H_preds_masked  = [H_station - denorm_pred_masked for denorm_pred_masked in denorm_preds_masked]
    
    # Weights of the masks (based on predictions from perturbed instances)
    weights = np.concatenate(denormalized_H_preds_masked, axis=0)

    ### S1 - Compute saliency map
    s1_i = calculate_saliency_map_fn(N, weights, masks)
    
    ### S2 - Compute saliency map with conditional variance
    s2_i = calculate_saliency_map_ev_masks_cond_var(N, weights, s1_i, masks)
    
    ### S3 - Compute saliency map with conditional bias
    s3_i = calculate_saliency_map_ev_masks_cond_bias(N, weights, denormalized_H_pred_original, masks)
    
    ### S4 - Compute saliency map with RMSE
    s4_i = np.sqrt(calculate_saliency_map_ev_masks_cond_var(N, weights, denormalized_H_pred_original, masks))

    print(f"############### Processo completato. Mappa di salienza generata per Istanza #{nr_instance} ###############")

    return np.squeeze(s1_i), np.squeeze(s2_i), np.squeeze(s3_i), np.squeeze(s4_i)

"""#### ***Evaluation Metrics***"""

def calculate_auc(x, y):
    """
    Calculates the area under the curve (AUC) using the trapezoidal method.

    :param x: X-axis values (fraction of inserted pixels/frames).
    :param y: Y-axis values (calculated errors).
    :return: Area under the curve.
    """
    return np.trapz(y, x)


# Returns the nth percentile of the most important pixels of the input saliency map
def get_top_n_pixels(saliency_map, n):
    # Flatten the saliency map
    flat_saliency = saliency_map.flatten()
    # Sort the indices of the elements in descending order of saliency
    sorted_indices = np.argsort(flat_saliency)[::-1]

    # Calculate the number of columns of the saliency map
    num_cols = saliency_map.shape[1]

    top_pixels = []
    for i in range(n):
        idx = sorted_indices[i]
        row, col = divmod(idx, num_cols)
        top_pixels.append((row, col))

    return top_pixels

"""##### ***Insertion***"""

def update_instance_with_pixels(current_instance, original_instance, x, y):
    """
    Updates the image by inserting the most important pixels.

    :param current_instance: Current instance.
    :param original_instance: Original instance.
    :param x: x-coordinate of the pixel to insert.
    :param y: y-coordinate of the pixel to insert.
    :return: Updated instance with the superpixel.
    """
    new_current_instance = current_instance.copy()
    new_current_instance[:, x, y, 0] = original_instance[:, x, y, 0]

    return new_current_instance


def insertion(model, original_instance, x3_instance, sorted_per_importance_pixels_index, initial_blurred_instance, original_prediction, H_station=390.0):
    """
    Calculates the insertion metric for a given explanation.

    :param model: Black-box model.
    :param original_instance: Original instance.
    :param sorted_per_importance_pixels_index: List of lists of all superpixels sorted by importance.
    :param initial_blurred_images: Initial image with all pixels set to zero.
    :return: List of errors at each insertion step.
    """

    # List to store instances to which pixels are gradually added. Initialized with the blurred initial instance.
    insertion_images = [initial_blurred_instance]

    # Prediction on the initial image (all pixels set to zero)
    I_prime = copy.deepcopy(initial_blurred_instance)

    # Gradually add the most important pixels (for each frame). I get a list of images with pixels added gradually.
    for x, y in sorted_per_importance_pixels_index:
        I_prime = update_instance_with_pixels(I_prime, original_instance, x, y)
        insertion_images.append(I_prime)

    insertion_images = [img.astype(np.float32) for img in insertion_images]
    # Calculate predictions on the instances where pixels were gradually added
    new_predictions = ensemble_predict(model, insertion_images, x3_instance)
    denorm_new_predictions = [(new_prediction * vott_target_test_std) + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]

    # Calculate MSE compared to the original instance's prediction (as in the test-set). Ignore the first one which is for the blurred image.
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions[1:]]

    initial_error = mean_squared_error(original_prediction, denormalized_H_new_predictions[0])
    print(f"Initial Prediction with Blurred Instance. Prediction: {denormalized_H_new_predictions[0]}, error: {initial_error}")
    only_inserted_pixel_new_predictions = denormalized_H_new_predictions[1:]

    total_errors = [initial_error] + errors  # Initial error + errors for all inserted pixels

    # New X-axis
    x = np.linspace(0, 1, len(total_errors))
    # Calculate AUC with the new x-axis
    auc = calculate_auc(x, total_errors)
    print(f"Area under the curve (AUC): {auc}")
    return total_errors, auc

def update_image_by_removing_pixels(current_instance, x, y, std_zero_value=-0.6486319166678826):
    """
    Updates the image by removing the specified x, y pixels.

    :param current_instance: Current instance.
    :param x: x-coordinate of the pixel to remove.
    :param y: y-coordinate of the pixel to remove.
    :return: Updated instance with the x, y removed for all time-steps.
    """
    new_instance = copy.deepcopy(current_instance)
    new_instance[:, x, y, 0] = std_zero_value  # Set the pixel to normalized zero for Prec.
    return new_instance

def deletion(models, original_instance, x3_instance, sorted_per_importance_pixels_index, original_prediction, H_station=390.0):
    """
    Calculates the removal metric for a given explanation.

    :param models: List of pre-trained models.
    :param original_instance: Original image.
    :param x3_instance: One-hot encoding for the prediction.
    :param sorted_per_importance_pixels_index: Indices of pixels in order of importance.
    :return: List of errors and AUC at each removal step.
    """
    # List to store images to which pixels are gradually removed (for each time-step)
    deletion_images = []

    # Initialization
    I_prime = copy.deepcopy(original_instance)

    # Gradually remove the most important pixels (for each frame). I get a list of images with pixels removed.
    for x, y in sorted_per_importance_pixels_index:
        I_prime = update_image_by_removing_pixels(I_prime, x, y)
        deletion_images.append(I_prime)

    # Calculate the prediction for all images where pixels were gradually removed.
    new_predictions = ensemble_predict(models, deletion_images, x3_instance)
    denorm_new_predictions = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    # Calculate MSE compared to the original prediction.
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions]

    initial_error = 0.0
    print(f"Initial Prediction with Original Images, prediction: {original_prediction}, error: {initial_error}")

    total_errors = [initial_error] + errors  # Initial error + errors for all removed pixels

    # Normalize the fraction of removed pixels.
    x = np.linspace(0, 1, len(total_errors))
    # Calculate the AUC
    auc = calculate_auc(x, total_errors)

    print(f"Area under the curve (AUC): {auc}")
    return total_errors, auc

""" Experiments """

channel_prec = 0
models = vott_lstm_models_loaded
seed = 42
T,H,W,C = (104,5,8,3)
std_zero_value = -0.6486319166678826
H_station = 390.0

N = 1000

h_w_values = [(1,2),(2,1),(2,2),(2,3),(2,4)]
p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

results_setups = []

len_test_set = len(vottignasco_test_image)

for h,w in h_w_values:
    #print(f"############################## Setup h,w: ({h},{w}) ##############################")
    for p in p_values:
        # Save all sal_maps for the entire Test-Set
        saliency_maps = np.zeros((len_test_set,4,H,W))
        # Errors and AUC Insertion/Deletion for the entire Test-Set
        errors_insertion_all_testset = np.zeros((len_test_set,4, H*W+1))
        auc_insertion_all_testset    = np.zeros((len_test_set,4, 1))
        errors_deletion_all_testset  = np.zeros((len_test_set,4, H*W+1))
        auc_deletion_all_testset     = np.zeros((len_test_set,4, 1))

        p_str = str(p).replace(".", "")
        param_combination = f"h{h}_w{w}_p{p_str}"
        print(f"############################## Setup - > parameters combination: {param_combination} ##############################")  

        execution_times = []
        
        for nr_instance,_ in enumerate(vottignasco_test_image):
            print(f"###################### Explanation for Instance #{nr_instance} ####################################")
            time_start = datetime.datetime.now()

            s1_i,s2_i,s3_i,s4_i = rise_spatial_explain(nr_instance, vottignasco_test_image, vottignasco_test_OHE, models, channel_prec,
                                                        N, generate_rise_masks_2d, seed, multiplicative_uniform_noise_onechannel, calculate_saliency_map_ev_masks, H_station, h=h, w=w, p=p)
            
            time_end = datetime.datetime.now()
            exec_time = (time_end - time_start).total_seconds()
            
            execution_times.append(exec_time)
           
            # DEBUG
            print("s1 frame 0:", s1_i[0],"\n")
            print("s2 frame 0:", s2_i[0], "\n")
            print("s3 frame 0:", s3_i[0], "\n")
            print("s4 frame 0:", s4_i[0], "\n")

            saliency_maps[nr_instance][0] = s1_i
            saliency_maps[nr_instance][1] = s2_i
            saliency_maps[nr_instance][2] = s3_i
            saliency_maps[nr_instance][3] = s4_i

            instance    = copy.deepcopy(vottignasco_test_image[nr_instance])
            x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])

            # Insertion s2,s3,s4
            # Blurred video to start from for the Insertion. All Prec pixels set to std_zero_value
            initial_blurred_instance = copy.deepcopy(instance)
            initial_blurred_instance[:,:,:,channel_prec] = std_zero_value

            original_instance = copy.deepcopy(instance)
            original_prediction = ensemble_predict(models, original_instance, x3_instance)
            denormalized_H_original_prediction = H_station - (original_prediction * vott_target_test_std + vott_target_test_mean)
            print(f"Original Prediction: {denormalized_H_original_prediction}")

            all_important_pixels_s2 = get_top_n_pixels(s2_i, instance.shape[1]*instance.shape[2])[::-1]
            all_important_pixels_s3 = get_top_n_pixels(np.abs(s3_i), instance.shape[1]*instance.shape[2])[::-1]
            all_important_pixels_s4 = get_top_n_pixels(s4_i, instance.shape[1]*instance.shape[2])[::-1]

                     
            errors_insertion_s2,auc_insertion_s2 = insertion(models, original_instance, x3_instance, all_important_pixels_s2, initial_blurred_instance, denormalized_H_original_prediction) # s2   
            errors_insertion_s3,auc_insertion_s3 = insertion(models, original_instance, x3_instance, all_important_pixels_s3, initial_blurred_instance, denormalized_H_original_prediction) # s3
            errors_insertion_s4,auc_insertion_s4 = insertion(models, original_instance, x3_instance, all_important_pixels_s4, initial_blurred_instance, denormalized_H_original_prediction) # s4
            #print(f"Errors Insertion: {errors_insertion}")
            #print(f"AUC Insertion: {auc_insertion}")

            for nr_error in range (0, (H*W+1)):
                errors_insertion_all_testset[nr_instance][1][nr_error] = errors_insertion_s2[nr_error]
                errors_insertion_all_testset[nr_instance][2][nr_error] = errors_insertion_s3[nr_error]
                errors_insertion_all_testset[nr_instance][3][nr_error] = errors_insertion_s4[nr_error]
            #for nr_error, error in enumerate(errors_insertion):
            #    errors_insertion_all_testset[nr_instance][nr_error] = error
            
            auc_insertion_all_testset[nr_instance][1] = auc_insertion_s2
            auc_insertion_all_testset[nr_instance][2] = auc_insertion_s3
            auc_insertion_all_testset[nr_instance][3] = auc_insertion_s4

            # Deletion
            errors_deletion_s2,auc_deletion_s2 = deletion(models, original_instance, x3_instance, all_important_pixels_s2, denormalized_H_original_prediction) # s2
            errors_deletion_s3,auc_deletion_s3 = deletion(models, original_instance, x3_instance, all_important_pixels_s3, denormalized_H_original_prediction) # s3
            errors_deletion_s4,auc_deletion_s4 = deletion(models, original_instance, x3_instance, all_important_pixels_s4, denormalized_H_original_prediction) # s4
            #print(f"Errors Deletion: {errors_deletion}")
            #print(f"AUC Deletion: {auc_deletion}")

            for nr_error in range (0, (H*W+1)):
                errors_deletion_all_testset[nr_instance][1][nr_error] = errors_deletion_s2[nr_error]
                errors_deletion_all_testset[nr_instance][2][nr_error] = errors_deletion_s3[nr_error]
                errors_deletion_all_testset[nr_instance][3][nr_error] = errors_deletion_s4[nr_error]
            
            auc_deletion_all_testset[nr_instance][1] = auc_deletion_s2
            auc_deletion_all_testset[nr_instance][2] = auc_deletion_s3
            auc_deletion_all_testset[nr_instance][3] = auc_deletion_s4

        print(f"#################################### END for all Instance in Test-Set for {param_combination} ####################################")

        result = {
                "saliency_maps": saliency_maps,
                "errors_insertion": errors_insertion_all_testset,
                "auc_insertion": auc_insertion_all_testset,
                "errors_deletion": errors_deletion_all_testset,
                "auc_deletion": auc_deletion_all_testset,
                "parameters_comb": param_combination,
                "execution_times": execution_times  # List of nr_instances (105) execution times for each instance
            }

        results_setups.append(result)

        path_to_save_results = f"{RESULT_DIR}/###### REPLACE WITH REAL PATH #######/rise_original_spatial_results_all_setup.pkl"
        # Save the results list to a pickle file
        with open(path_to_save_results, 'wb') as f:
            pickle.dump(results_setups, f)

        print(f"############################# END FOR ALL INSTANCES ###############################################")
print("############################# END FOR ALL SETUPS ##########################################################################")

