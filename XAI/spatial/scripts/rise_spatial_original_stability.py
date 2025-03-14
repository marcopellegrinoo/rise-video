# -*- coding: utf-8 -*-
"""rise_spatial_multiplicative_norm_zero_cineca.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-4UndkKCTVbDRQZql3km54NZicEG05gb

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

# Ottieni il percorso effettivo da una variabile d'ambiente
#work_path = os.environ['WORK']  # Ottieni il valore della variabile d'ambiente WORK
#v_test_OHE_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_month_OHE.npy")
v_test_OHE_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_month_OHE.npy"
#v_test_image_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_normalized_image_sequences.npy")
v_test_image_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_normalized_image_sequences.npy"
#v_test_target_dates_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_target_dates.npy")
v_test_target_dates_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_target_dates.npy"
#v_test_normalization_factors_std_path  = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/normalization_factors/Vottignasco_00425010001_training_target_std.npy")
v_test_normalization_factors_std_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/normalization_factors/Vottignasco_00425010001_training_target_std.npy"
#v_test_normalization_factors_mean_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/normalization_factors/Vottignasco_00425010001_training_target_mean.npy")
v_test_normalization_factors_mean_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/normalization_factors/Vottignasco_00425010001_training_target_mean.npy"

# Carica l'array numpy dai file
vottignasco_test_OHE    = np.load(v_test_OHE_path)
vottignasco_test_image  = np.load(v_test_image_path)
vottignasco_test_dates  = np.load(v_test_target_dates_path)
vott_target_test_std    = np.load(v_test_normalization_factors_std_path) 
vott_target_test_mean   = np.load(v_test_normalization_factors_mean_path)

print(len(vottignasco_test_dates))
print(len(vottignasco_test_image))
print(len(vottignasco_test_OHE))

#print(vottingasco_test_OHE[0], "\n")
#print(vottignasco_test_image[0][0], "\n")

# """##### ***Black Boxes***""

# Se vuoi abilitare il dropout a runtime
mc_dropout = True

# Definizione della classe personalizzata doprout_custom
class doprout_custom(tf.keras.layers.SpatialDropout1D):
    def call(self, inputs, training=None):
        if mc_dropout:
            return super().call(inputs, training=True)
        else:
            return super().call(inputs, training=False)

# Percorso della directory su Cineca
#base_dir = os.path.join(os.environ['WORK'], "Water_Resources/rise-video/trained_models/seq2val/Vottignasco")
base_dir  = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/trained_models/seq2val/Vottignasco"
lstm_suffix = 'time_dist_LSTM'

vott_lstm_models = []

def extract_index(filename):
    """Funzione per estrarre l'indice finale dal nome del file."""
    return int(filename.split('_LSTM_')[-1].split('.')[0])

# Trova tutti i file .keras nella cartella e li aggiunge alla lista
for filename in os.listdir(base_dir):
    if lstm_suffix in filename and filename.endswith(".keras"):
        vott_lstm_models.append(os.path.join(base_dir, filename))

# Ordina i modelli in base all'indice finale
vott_lstm_models = sorted(vott_lstm_models, key=lambda x: extract_index(os.path.basename(x)))

# Lista per i modelli caricati
vott_lstm_models_loaded = []

for i, model_lstm_path in enumerate(vott_lstm_models[:10]):  # Prendo i primi 10 modelli ordinati
    #print(f"Caricamento del modello LSTM {i+1}: {model_lstm_path}")

    # Carico il modello con la classe custom
    model = load_model(model_lstm_path, custom_objects={"doprout_custom": doprout_custom})

    # Aggiungo il modello alla lista
    vott_lstm_models_loaded.append(model)

print(vott_lstm_models_loaded)

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_rise_masks_2d(N, input_size, seed, **kwargs):
    """
    Genera N maschere RISE per un'immagine di dimensioni HxW.

    Parametri:
    - N: numero di maschere
    - input_size: (H,W) dimensioni dell'immagine originale
    - h, w: dimensioni delle maschere a bassa risoluzione
    - p: probabilità di attivazione dei pixel nella maschera binaria iniziale

    Ritorna:
    - masks: array di shape (N, H, W) contenente le maschere normalizzate.
    """

    h  = kwargs.get("h", 2)
    w = kwargs.get("w", 2)
    p = kwargs.get("p", 0.5)

    np.random.seed(seed)

    masks = []
    H,W = input_size
    CH, CW = H // h, W // w  # Fattore di upscaling

    for _ in range(N):
        # 1. Generazione della maschera binaria iniziale (h x w)
        small_mask = np.random.rand(h, w) < p

        up_size_h = (h+1) * CH
        up_size_w = (w+1) * CW

        # 2. Upsampling bilineare alla dimensione (H + CH, W + CW
        upsampled_mask = cv2.resize(small_mask.astype(np.float32),
                                    (up_size_w, up_size_h), interpolation=cv2.INTER_LINEAR)
        
        #print(upsampled_mask.shape)
        
        # 3. Crop casuale della regione H x W
        x_offset = np.random.randint(0, (up_size_h - H) + 1)
        y_offset = np.random.randint(0, (up_size_w - W) + 1)
        final_mask = upsampled_mask[x_offset:x_offset + H, y_offset:y_offset + W] 

        #print(final_mask.shape)

        masks.append(final_mask)

    #masks = np.array(masks)  # Converte la lista in array NumPy
    #masks = masks[~(masks == 0).all(axis=(1, 2))]  # Filtro maschere vuote
    #masks = masks[~(masks == 1.0).all(axis=(1, 2))]  # Filtro maschere con tutti 1.0
    return np.array(masks)


def multiplicative_uniform_noise_onechannel(images, masks, channel, **kwargs):
    std_zero_value = kwargs.get("std_zero_value", -0.6486319166678826)

    masked = []

    # Itero su tutte le N maschere generate
    for mask in masks:
        masked_images = copy.deepcopy(images)  # Copia profonda delle immagini originali

        # Applica la perturbazione solo al canale specificato
        masked_images[..., channel] = (
            masked_images[..., channel] * mask + (1 - mask) * std_zero_value)

        masked.append(masked_images)

    return masked

def ensemble_predict(models, images, x3_exp, batch_size=1000):
    # Assicuriamoci che images sia una lista
    if not isinstance(images, list):
        images = [images]

    len_x3 = len(images)

    # Convertiamo x3_exp in un tensore replicato per ogni immagine
    x3_exp_tensor = tf.convert_to_tensor(x3_exp, dtype=tf.float32)

    # Lista per raccogliere le predizioni finali
    final_preds = []

    # Processamento a batch
    for i in range(0, len_x3, batch_size):
        batch_images = images[i:i + batch_size]
        batch_len = len(batch_images)

        # Conversione batch in tensori
        Y_test = tf.stack([tf.convert_to_tensor(img, dtype=tf.float32) for img in batch_images])
        Y_test_x3 = tf.tile(tf.expand_dims(x3_exp_tensor, axis=0), [batch_len, 1, 1])

        # Raccoglie le predizioni di tutti i modelli per il batch corrente
        batch_preds = []

        for model in models:
            preds = model.predict([Y_test, Y_test_x3], verbose=0)
            batch_preds.append(preds)

        # Converte le predizioni del batch in un tensore e calcola la media
        batch_preds_tensor = tf.stack(batch_preds)
        mean_batch_preds = tf.reduce_mean(batch_preds_tensor, axis=0)

        # Aggiunge le predizioni del batch alla lista finale
        final_preds.extend(mean_batch_preds.numpy())

    return np.array(final_preds)

"""#### ***Saliency Map***"""

def calculate_saliency_map_ev_masks(N, weights, masks):
    """
    Calcola la mappa di salienza media data una serie di predizioni e maschere.

    :param weights_array: Array di predizioni (pesi delle maschere).
    :param masks: Array di maschere (numero di maschere x dimensioni maschera).
    :return: Mappa di salienza media.
    """
    sal = []
    
    for j in range(len(masks)):
        sal_j = weights[j] * masks[j]
        sal.append(sal_j)

    # Ora calcola la media lungo l'asse 0
    ev_masks = np.mean(masks, axis=0)

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # aggiunta della frazione 1/valore_atteso(maschere)

    return np.squeeze(sal)

def calculate_saliency_map_ev_masks_cond_var(N, weights, ref, masks):
    """
    Calcola la mappa di salienza media data una serie di predizioni e maschere.
    VARIANZA CONDIZIONATA

    :param weights_array: Array di predizioni (pesi delle maschere).
    :param masks: Array di maschere (numero di maschere x dimensioni maschera).
    :return: Mappa di salienza media.
    """
    sal = []
    
    for j in range(len(masks)):
        sal_j = ((weights[j] - ref)**2) * masks[j]
        sal.append(sal_j)

    # Ora calcola la media lungo l'asse 0
    ev_masks = np.mean(masks, axis=0)

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # aggiunta della frazione 1/valore_atteso(maschere)

    return np.squeeze(sal)

def calculate_saliency_map_ev_masks_cond_bias(N, weights, ref, masks):
    """
    Calcola la mappa di salienza media data una serie di predizioni e maschere.
    VARIANZA CONDIZIONATA

    :param weights_array: Array di predizioni (pesi delle maschere).
    :param masks: Array di maschere (numero di maschere x dimensioni maschera).
    :return: Mappa di salienza media.
    """
    sal = []
    
    for j in range(len(masks)):
        sal_j = (weights[j] - ref) * masks[j]
        sal.append(sal_j)

    # Ora calcola la media lungo l'asse 0
    ev_masks = np.mean(masks, axis=0)

    sal = (1/(ev_masks * N)) * np.sum(sal, axis=0)  # aggiunta della frazione 1/valore_atteso(maschere)

    return np.squeeze(sal)

"""#### ***Spatial-RISE: Framework***"""

def rise_spatial_explain(nr_instance, data_test_image, data_test_OHE, models, channel,
                          N, generate_masks_fn, seed, perturb_instance_fn, calculate_saliency_map_fn, H_station=390.0, **kwargs):
  print(f"############################### RISE-Spatial on Instance #{nr_instance} ###########################")
  instance    = copy.deepcopy(data_test_image[nr_instance])
  x3_instance = copy.deepcopy(data_test_OHE[nr_instance])

  input_size = (instance.shape[1], instance.shape[2])

  masks = generate_masks_fn(N, input_size, seed, **kwargs)
  perturbed_instances = perturb_instance_fn(instance, masks, channel)

  # Predizione su Istanza Originale
  pred_original = ensemble_predict(models, instance, x3_instance)
  # Predizioni su Istanze Perturbate
  preds_masked = ensemble_predict(models, perturbed_instances, x3_instance)

  # Denormalizzazione Output Black-Box con H_station 
  denorm_pred_original = (pred_original * vott_target_test_std) + vott_target_test_mean
  denorm_preds_masked  = [(pred_masked * vott_target_test_std) + vott_target_test_mean for pred_masked in preds_masked]
  denormalized_H_pred_original = H_station - denorm_pred_original
  denormalized_H_preds_masked  = [H_station - denorm_pred_masked for denorm_pred_masked in denorm_preds_masked]
  # Pesi delle Maschere
  weights = np.concatenate(denormalized_H_preds_masked, axis=0)

  ### S1 
  s1_i = calculate_saliency_map_fn(N, weights, masks)
  ### S2
  s2_i = calculate_saliency_map_ev_masks_cond_var(N, weights, s1_i, masks)
  ### S3 (BIAS)
  s3_i = calculate_saliency_map_ev_masks_cond_bias(N, weights, denormalized_H_pred_original, masks)
  ### S4 (RMSE)
  s4_i = np.sqrt(calculate_saliency_map_ev_masks_cond_var(N, weights, denormalized_H_pred_original, masks))
  print(f"############### Processo completato. Mappa di salienza generata per Istanza #{nr_instance} ###############")

  return np.squeeze(s1_i), np.squeeze(s2_i), np.squeeze(s3_i), np.squeeze(s4_i)

"""#### ***Evaluation Metrics***"""

def calculate_auc(x, y):
    """
    Calcola l'area sotto la curva (AUC) utilizzando il metodo del trapezio.

    :param x: Valori dell'asse x (frazione dei pixel/frame inseriti).
    :param y: Valori dell'asse y (errori calcolati).
    :return: Area sotto la curva.
    """
    return np.trapz(y, x)


# Restituisce n-esimo percentile dei pixel più importanti delle mappa di salienza data in input
def get_top_n_pixels(saliency_map, n):
    # Appiattisci la mappa di salienza
    flat_saliency = saliency_map.flatten()
    # Ordina gli indici degli elementi in ordine decrescente di salienza
    sorted_indices = np.argsort(flat_saliency)[::-1]

    # Calcola il numero di colonne della mappa di salienza
    num_cols = saliency_map.shape[1]

    top_pixels = []
    for i in range(n):
        idx = sorted_indices[i]
        row, col = divmod(idx, num_cols)
        top_pixels.append((row, col))

    return top_pixels

"""##### ***Insertion***"""

def update_instance_with_pixels(current_instance, original_instance, x,y):
    """
    Aggiorna l'immagine inserendo i pixel più importanti.

    :param current_instance: Istanza corrente.
    :param original_instance: Istanza originale.
    :param x: coordinata x del pixel da inserire
    :param y: coordinata y del pixel da inserire
    :return: Istanza aggiornata con il superpixel.
    """
    new_current_instance = current_instance.copy()
    new_current_instance[:, x, y, 0] = original_instance[:, x, y, 0]

    return new_current_instance


def insertion(model, original_instance, x3_instance, sorted_per_importance_pixels_index, initial_blurred_instance, original_prediction, H_station=390.0):
    """
    Calcola la metrica di inserimento per una spiegazione data.

    :param model: Black-box.
    :param original_instance: Istanza originale.
    :param sorted_per_importance_pixels_index: Lista di liste di tutti i superpixel per importanza
    :param initial_blurred_images: Immagine iniziale con tutti i pixel a zero.
    :return: Lista degli errori ad ogni passo di inserimento.
    """

    # Lista per memorizzare le istanze a cui aggiungo pixel mano a mano. Inizializzata con istanza iniziale blurrata
    insertion_images = [initial_blurred_instance]

    # Predizione sull'immagine iniziale (tutti i pixel a zero)
    I_prime = copy.deepcopy(initial_blurred_instance)

    # Aggiungere gradualmente i pixel (per ogni frame) più importanti. Ottengo una lista con tutte le img con i pixel aggiunti in maniera graduale
    for x,y in sorted_per_importance_pixels_index:
        I_prime = update_instance_with_pixels(I_prime, original_instance, x,y)
        insertion_images.append(I_prime)

    insertion_images = [img.astype(np.float32) for img in insertion_images]
    # Calcolo le predizioni sulle istanze a cui ho aggiunto i pixel in maniera graduale
    new_predictions = ensemble_predict(model, insertion_images, x3_instance)
    denorm_new_predictions  = [(new_prediction * vott_target_test_std)+ vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions  = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]

    # Rispetto ad ogni suddetta predizione, calcolo il MSE rispetto la pred sull'istanza originaria (come da test-set). Ignora la prima che è sull'img blurrata originale
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions[1:]]

    initial_error = mean_squared_error(original_prediction, denormalized_H_new_predictions[0])
    print(f"Initial Prediction with Blurred Instance. Prediction: {denormalized_H_new_predictions[0]}, error: {initial_error}")
    only_inserted_pixel_new_predictions = denormalized_H_new_predictions[1:]

    #for nr_pixel, error in enumerate(errors):
    #  print(f"Inserted Pixel: {sorted_per_importance_pixels_index[nr_pixel]}. Prediction: {only_inserted_pixel_new_predictions[nr_pixel]}, error: {error}")

    total_errors = [initial_error] + errors # Errore iniziale + errori su tutti i pixel inseriti

    # # Nuovo asse X
    x = np.linspace(0, 1, len(total_errors))
    # Calcolo dell'AUC con il nuovo asse x
    auc = calculate_auc(x, total_errors)
    print(f"Area under the curve (AUC): {auc}")
    return total_errors,auc

def update_image_by_removing_pixels(current_instance, x, y, std_zero_value=-0.6486319166678826):
    """
    Aggiorna l'immagine rimuovendo i pixel x,y indicati.

    :param current_instance: istanza corrente.
    :param x: coordinata x del pixel da rimuovere
    :param y: coordinata y del pixel da rimuovere
    :return: Istanza aggiornata con x,y rimossi su tutti time-step.
    """
    new_instance = copy.deepcopy(current_instance)
    new_instance[:, x, y, 0] = std_zero_value # Imposta i pixel a zero normalizzato per Prec
    return new_instance

def deletion(models, original_instance, x3_instance, sorted_per_importance_pixels_index, original_prediction, H_station=390.0):
    """
    Calcola la metrica di rimozione per una spiegazione data.

    :param models: Lista di modelli pre-addestrati.
    :param original_instance: Immagine originale.
    :param x3_instance: Codifica one-hot per la previsione.
    :param sorted_per_importance_pixels_index: Indici dei pixel in ordine di importanza.
    :return: Lista degli errori, auc ad ogni passo di rimozione.
    """
    # Lista per memorizzare le img a cui elimino gradualmente i pixels (per ogni time-step)
    deletion_images = []

    # Inizializzazione
    I_prime = copy.deepcopy(original_instance)

    # Aggiungere gradualmente i pixel (per ogni frame) più importanti. Ottengo una lista con tutte le img con i pixel rimossi
    for x, y in sorted_per_importance_pixels_index:
        I_prime = update_image_by_removing_pixels(I_prime, x, y)
        deletion_images.append(I_prime)

    # Calcolo della predizione su tutte le img a cui ho rimosso gradualmente i pixel
    new_predictions = ensemble_predict(models, deletion_images, x3_instance)
    denorm_new_predictions  = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions  = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    # Calcolo del mse rispetto la predizione originale
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions]

    initial_error = 0.0
    print(f"Initial Prediction with Original Images, prediction: {original_prediction}, error: {initial_error}")
    #for nr_pixel, error in enumerate(errors):
    #    print(f"Removed pixel {sorted_per_importance_pixels_index[nr_pixel]}, new prediction: {denormalized_H_new_predictions[nr_pixel]}, error: {error}")

    total_errors = [initial_error] + errors  # Errore iniziale + errori su tutti i pixel rimossi

    # Normalizzare la frazione di pixel rimossi
    x = np.linspace(0, 1, len(total_errors))
    # Calcolo dell'AUC
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

N = 1000

h, w = 2,4    #"???"   # Valori fissi per h e w (CONFIGURAZIONE MIGLIORE)
p = 0.5       #"???"      # Valore fisso per p (CONFIGURAZIONE MIGLIORE)

M = 10        ####"???" # Numero di seed casuali da generare
seed_values = np.random.choice(100, M, replace=False)

results_setups = []

len_test_set = len(vottignasco_test_image)

for seed in seed_values:
    param_combination = f"seed{seed}"

    print(f"############################## Setup - > parameters combination: {param_combination} ##############################")  

    # Conserva tutte le sal_maps per tutto il Test-Set
    saliency_maps = np.zeros((len_test_set,4,H,W))
    # Errori e AUC Insertion/Deletion tutto il Test-Set
    errors_insertion_all_testset = np.zeros((len_test_set,4, H*W+1))
    auc_insertion_all_testset    = np.zeros((len_test_set,4, 1))
    errors_deletion_all_testset  = np.zeros((len_test_set,4, H*W+1))
    auc_deletion_all_testset     = np.zeros((len_test_set,4, 1))

    execution_times = []
    
    for nr_instance,_ in enumerate(vottignasco_test_image):
        print(f"###################### Explanation for Instance #{nr_instance} ####################################")
        time_start = datetime.datetime.now()

        s1_i,s2_i,s3_i,s4_i = rise_spatial_explain(nr_instance, vottignasco_test_image, vottignasco_test_OHE, models, channel_prec,
                                                    N, generate_rise_masks_2d, seed, multiplicative_uniform_noise_onechannel, calculate_saliency_map_ev_masks, H_station, h=h, w=w, p=p)
        
        time_end = datetime.datetime.now()
        exec_time = (time_end - time_start).total_seconds()
        
        execution_times.append(exec_time)

        saliency_maps[nr_instance][0] = s1_i
        saliency_maps[nr_instance][1] = s2_i
        saliency_maps[nr_instance][2] = s3_i
        saliency_maps[nr_instance][3] = s4_i

        instance    = copy.deepcopy(vottignasco_test_image[nr_instance])
        x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])

        # Insertion s2,s3,s4
        # Video blurrato da cui partire per l'Insertion. Tutti i pixel di Prec su std_zero_value
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

        for nr_error in range (0, (H*W+1)):
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
            "execution_times": execution_times
        }

    results_setups.append(result)

    path_to_save_results = f"{RESULT_DIR}/rise_spatial_original_stability_{param_combination}.pkl"
    with open(path_to_save_results, 'wb') as f:
        pickle.dump(results_setups, f)

print("############################# END FOR ALL SETUPS ##########################################################################")
