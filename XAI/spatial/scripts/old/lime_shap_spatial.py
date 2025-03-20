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
v_test_OHE_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_month_OHE.npy")
v_test_image_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_normalized_image_sequences.npy")
v_test_target_dates_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_target_dates.npy")
shapefile_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/shapefile_raster/")
v_test_normalization_factors_std_path = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/normalization_factors/Vottignasco_00425010001_training_target_std.npy")
v_test_normalization_factors_mean_path     = os.path.join(work_path, "Water_Resources/rise-video/data/Vottignasco/normalization_factors/Vottignasco_00425010001_training_target_mean.npy")

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

# """##### ***Black Boxes***"""


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
base_dir = os.path.join(os.environ['WORK'], "Water_Resources/rise-video/trained_models/seq2val/Vottignasco")
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

"""### ***Spatial-LIME***

#### ***Spatial-Superpixels***
"""

def create_spatial_superpixels(shapefile_path, n_segments=8, compactness=15):
  # DTM [50m] import
  dtm_piemonte = rioxarray.open_rasterio(shapefile_path + 'DTMPiemonte_filled_50m.tif')
  dtm_piemonte = dtm_piemonte.rio.reproject("epsg:4326")
  dtm_piemonte = dtm_piemonte.where(dtm_piemonte != -99999) # Take valid pixel

  # Catchment shapefile
  catchment = gpd.read_file(shapefile_path + "BAC_01_bacialti.shp") # select GRANA-MAIRA	and VARAITA
  catchment = catchment.to_crs('epsg:4326')

  # Select only the Grana-Maira catchment
  catchment_GM = catchment.loc[catchment.NOME == "GRANA-MAIRA"]
  catchment_GM = catchment_GM.reset_index(drop = True)

  # Retrieve the borders of the catchment from the shapefile
  xmin_clip, ymin_clip, xmax_clip, ymax_clip = catchment_GM.total_bounds
  # Extend the borders to include more pixel on the borders

  increase = 0.05 # Degrees
  #ymin_clip -= increase # not needed
  xmin_clip += increase # "+" for subset for pixel included in the mask
  xmax_clip += increase
  #ymax_clip += increase # not needed

  dtm_piemonte_clipped = dtm_piemonte.rio.clip_box( minx = xmin_clip, maxx= xmax_clip , miny= ymin_clip , maxy= ymax_clip)

  # Creazione img 5x8 cone lat,lon,dtm
  # Definizione delle coordinate
  lon = np.array([6.938, 7.063, 7.188, 7.313, 7.438, 7.563, 7.688, 7.813])  # 8 valori
  lat = np.array([44.313, 44.438, 44.563, 44.688, 44.813])  # 5 valori

  # Creazione di una griglia lat-lon 5x8
  lon_grid, lat_grid = np.meshgrid(lon, lat)

  # Creazione di un array 5x8x3
  img = np.zeros((5, 8, 3))

  # Assegno le coordinate nei primi due canali
  img[:, :, 0] = lat_grid  # Canale 0 = latitudine
  img[:, :, 1] = lon_grid  # Canale 1 = longitudine
  img[:, :, 2] = 0  # Canale 2 = valore placeholder

  for nr_lat,latitude in enumerate(lat):
    for  nr_lon,longitude in enumerate(lon):
      img[nr_lat, nr_lon, 2] = dtm_piemonte_clipped.sel(x=longitude, y=latitude, method='nearest').values

  img = np.nan_to_num(img, nan=0.0)

  # SLIC
  segments = slic(img, n_segments=n_segments, compactness=compactness)

  # Creazione Spatial-Superpixels
  # Trova i valori unici nella matrice (i cluster)
  clusters = np.unique(segments)

  # Creazione di una lista di matrici binarie per ogni cluster
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

"""#### ***Rappresentazioni Interpretabili***"""

# Creazione dei z'

import itertools
import numpy as np

def generate_z_primes(n):
  """Args
      n (int): nr di superpixel considerati
     Return
      np.array: tutte le combinazioni possibili di 0 e 1 di lunghezza n
  """

  # Genera tutte le combinazioni possibili di 0 e 1 di lunghezza n
  z_primes = list(itertools.product([0, 1], repeat=n))
  # Converti le tuple in un array numpy (facoltativo)
  z_primes = np.array(z_primes)
  # Elimino il primo elemeno di tutti 0
  z_primes = z_primes[1:]
  # Elimino ultimo elemeno di tutti 1
  z_primes = z_primes[:-1]
  return z_primes

"""#### ***Generazione & Applicazione Maschere Rumore Uniforme (2D)***"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_masks(superpixels, input_size):
    N = len(superpixels)
    height, width = input_size
    masks = np.empty((N, height, width))

    for i in tqdm(range(N), desc='Generating masks'):
        mask = np.ones((height, width))
        indices = np.argwhere(superpixels[i] == 1)

        for (y, x) in indices:
            mask[y, x] = 0  # I pixel del cluster a 0

        masks[i] = mask

    return masks

"""#### ***Application Masks***"""

def multiplicative_uniform_noise_onechannel(instance, zs_primes, masks, channel, std_zero_value=-0.6486319166678826):
  """
  param:masks: maschere generate per ogni superpixel
  """
  masked = []
  for z in zs_primes:
    masked_instance = copy.deepcopy(instance)
    for i,z_i in enumerate(z):
      if z_i == 0:
         # Applica la perturbazione solo al canale specificato
        masked_instance[..., channel] = (
            masked_instance[..., channel] * masks[i] + (1 - masks[i]) * std_zero_value)

    masked.append(masked_instance)

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

"""#### ***Calcolo Weigths del Regressore***

###### ***LIME***
Dove *calculate_D*:
* D è la L2-Distance (Distanza Euclidea)
* x è l'istanza originaria da spiegare
* z è la versione perturbata non interpretabile
"""

def calculate_D(instance, perturbed_istance):
  x = instance.flatten()
  z = perturbed_istance.flatten()

  return np.linalg.norm(x - z)

def calculate_weigths_lime(instance, perturbed_instances, percentile_kernel_width):
  distances = [calculate_D(instance, perturbed_instance) for perturbed_instance in perturbed_instances]
  kernel_width = np.percentile(distances, percentile_kernel_width)
  # Importanza vicini
  weights = np.exp(- (np.array(distances) ** 2) / (kernel_width ** 2))
  return weights

"""##### ***Kernel-SHAP***"""

import math
from scipy.special import binom

def shap_kernel_weight(M, z):
  """
    Calcola il peso del kernel di Kernel SHAP per una data maschera (istanza interpretabile).

    Args:
        M (int): Numero totale di feature.
        z (array): Array contenente un zs_prime.

    Returns:
        float: Valore del kernel di pesatura di z'.
    """

  z_size = np.sum(z)
  #print("Mask size: ", mask_size)
  if z_size == 0 or z_size == M:
    return 0  # Peso nullo in questi casi estremi
  # Coefficiente binomiale: M su subset_size (|z'|)
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

def lime_shap(nr_instance, dataset_test_image, dataset_test_OHE, channels, models,
              n_segments, compactness, input_size, H_station=390.0, std_zero_value=-0.6486319166678826):

  height,width  = (input_size)
  channel_prec, channel_tmax, channel_min = channels

  instance    = copy.deepcopy(dataset_test_image[nr_instance])  # istanza da spiegare
  x3_instance = copy.deepcopy(dataset_test_OHE[nr_instance])    # One-Hot encode mesi dei frame dell'istanza

  # Creazione Superpixel Spaziali
  superpixels,_,_ = create_spatial_superpixels(shapefile_path, n_segments=n_segments, compactness=compactness)
  # Rappresentazioni Interpretabili dell'istanza
  zs_primes = generate_z_primes(len(superpixels))
  # Creazione maschere
  masks = generate_masks(superpixels, input_size)
  # Creazione dei vicini
  perturbed_instances = multiplicative_uniform_noise_onechannel(instance, zs_primes, masks, channel_prec, std_zero_value)
  # Predizione istanze perturbate
  preds_masked = ensemble_predict(models, list(perturbed_instances), x3_instance)
  # Denormalizzazione rispetto l'output della black-box
  denorm_preds_masked  = [pred_masked * vott_target_test_std + vott_target_test_mean for pred_masked in preds_masked]
  denormalized_H_preds_masked  = [H_station - denorm_pred_masked for denorm_pred_masked in denorm_preds_masked]

  return superpixels, zs_primes, perturbed_instances, denormalized_H_preds_masked

"""#### ***Evaluation Metrics***"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def calculate_auc(x, y):
    """
    Calcola l'area sotto la curva (AUC) utilizzando il metodo del trapezio.

    :param x: Valori dell'asse x (frazione dei pixel/frame inseriti).
    :param y: Valori dell'asse y (errori calcolati).
    :return: Area sotto la curva.
    """
    return np.trapz(y, x)

import numpy as np

def sorted_per_importance_superpixels_index(array):
    array = np.array(array)  # Converte in numpy array se non lo è già
    unique_values = np.unique(array)  # Trova i valori unici

    # Crea un dizionario con i valori come chiavi e liste di indici come valori
    indici_per_valore = {val: [] for val in unique_values}

    # Popola il dizionario con gli indici
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            indici_per_valore[array[i, j]].append((i, j))

    # Ordina i valori in ordine decrescente
    valori_ordinati = sorted(indici_per_valore.keys(), reverse=True)

    # Crea la lista finale con gli indici raggruppati per valore
    results = [indici_per_valore[val] for val in valori_ordinati]

    return results

"""##### ***Insertion***"""

def update_instance_with_superpixels(current_instance, original_instance, index_of_superpixels):
    """
    Aggiorna l'immagine inserendo i pixel più importanti.

    :param current_instance: Istanza corrente.
    :param original_instance: Istanza originale.
    :param index_of_superpixels: Lista contente gli indici del superpixel considerato
    :return: Istanza aggiornata con il superpixel.
    """
    new_current_instance = current_instance.copy()

    for x,y in index_of_superpixels:
      new_current_instance[:, x, y, 0] = original_instance[:, x, y, 0]
    return new_current_instance

def insertion(models, original_instance, x3, sorted_per_importance_all_superpixels_index, initial_blurred_instance, original_prediction, H_station=390.0):
    """
    Calcola la metrica di inserimento per una spiegazione data.

    :param models: Lista di modelli pre-addestrati.
    :param original_instance: Istanza originale.
    :param x3: Codifica one-hot per la previsione.
    :param sorted_per_importance_all_superpixels_index: Lista di liste di tutti i superpixel per importanza
    :param initial_blurred_images: Immagine iniziale con tutti i pixel a zero.
    :return: Lista degli errori ad ogni passo di inserimento.
    """

    # Lista per memorizzare le istanze a cui aggiungo pixel mano a mano. Inizializzata con istanza iniziale blurrata
    insertion_images = [initial_blurred_instance]

    # Predizione sull'immagine iniziale (tutti i pixel a zero)
    I_prime = copy.deepcopy(initial_blurred_instance)

    # Aggiungere gradualmente i pixel (per ogni frame) più importanti. Ottengo una lista con tutte le img con i pixel aggiunti in maniera graduale
    for index_of_superpixels in sorted_per_importance_all_superpixels_index:
        I_prime = update_instance_with_superpixels(I_prime, original_instance, index_of_superpixels)
        insertion_images.append(I_prime)

    # Calcolo le predizioni sulle istanze a cui ho aggiunto i pixel in maniera graduale
    new_predictions = ensemble_predict(models, insertion_images, x3)
    denorm_new_predictions  = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions  = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    # Rispetto ad ogni suddetta predizione, calcolo il MSE rispetto la pred sull'istanza originaria (come da test-set). Ignora la prima che è sull'img blurrata originale
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions[1:]]

    initial_error = mean_squared_error(original_prediction, denormalized_H_new_predictions[0])
    print(f"Initial Prediction with Blurred Instance, new prediction: {denormalized_H_new_predictions[0]}, error: {initial_error}")
    only_inserted_pixel_new_predictions = denormalized_H_new_predictions[1:]

    for nr_superpixel, error in enumerate(errors):
      print(f"SuperPixel: {sorted_per_importance_all_superpixels_index[nr_superpixel]}, new prediction: {only_inserted_pixel_new_predictions[nr_superpixel]}, error: {error}")

    total_errors = [initial_error] + errors # Errore iniziale + errori su tutti i pixel inseriti

    # Nuovo asse X: numero di superpixel inseriti (1, 2, ..., 8)
    x = np.arange(0, len(total_errors))  # Da 0 a 8 inclusi

    x_for_auc = np.linspace(0, 1, len(total_errors))
    # Calcolo dell'AUC con il nuovo asse x
    auc = calculate_auc(x_for_auc, total_errors)
    print(f"Area under the curve (AUC): {auc}")


    # # Plot della curva dell'errore e area sotto la curva (AUC)
    # plt.plot(x, total_errors, marker='o', linestyle='-', label='Error curve', color='blue')
    # # Pallini blu sui punti della curva
    # plt.scatter(x, total_errors, color='blue', zorder=3)

    # # Area sotto la curva
    # plt.fill_between(x, total_errors, color='skyblue', alpha=0.4)

    # # Testo AUC in alto a destra
    # plt.text(x[-1] * 0.95, max(total_errors) * 0.9, f'AUC: {auc:.2f}',
    #      horizontalalignment='right')

    # plt.xlabel('Number of superpixels inserted')  # Modifica etichetta asse X
    # plt.ylabel('Mean Squared Error')
    # plt.title('Insertion Metric Curve')
    # plt.xticks(x)  # Imposta i tick esattamente sui numeri interi (1, 2, ..., 8)
    # plt.legend()
    # #plt.grid(True, linestyle='--', alpha=0.6)  # Griglia più leggibile
    # plt.show()
    return total_errors,auc

"""##### ***Deletion***"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def update_instance_removing_superpixels(current_instance, index_of_superpixels, std_zero_value=-0.6486319166678826):
    """
    Aggiorna l'immagine inserendo i pixel più importanti.

    :param current_instance: Istanza corrente.
    :param original_instance: Istanza originale.
    :param index_of_superpixels: Lista contente gli indici del superpixel considerato
    :return: Istanza aggiornata con il superpixel.
    """
    new_current_instance = current_instance.copy()

    for x,y in index_of_superpixels:
      new_current_instance[:, x, y, 0] = std_zero_value
    return new_current_instance

def deletion(models, original_instance, x3_instance, sorted_per_importance_all_superpixels_index, original_prediction, H_station=390.0):
    """
    Calcola la metrica di deletion per una spiegazione data.

    :param models: Lista di modelli pre-addestrati.
    :param original_instance: Istanza originale.
    :param x3_instance: Codifica one-hot per la previsione.
    :param sorted_per_importance_all_superpixels_index: Lista di liste di tutti i superpixel per importanza
    :param original_prediction: Predizione originale.
    :return: Lista degli errori ad ogni passo di deletion.
    """

    # Lista per memorizzare le istanze a cui aggiungo pixel mano a mano. Inizializzata con istanza originale
    deletion_images = []

    # Predizione sull'immagine iniziale (tutti i pixel a zero)
    I_prime = copy.deepcopy(original_instance)

    # Aggiungere gradualmente i pixel (per ogni frame) più importanti. Ottengo una lista con tutte le img con i pixel aggiunti in maniera graduale
    for index_of_superpixels in sorted_per_importance_all_superpixels_index:
        I_prime = update_instance_removing_superpixels(I_prime, index_of_superpixels)
        deletion_images.append(I_prime)

    # Calcolo della predizione su tutte le img a cui ho rimosso gradualmente i pixel
    new_predictions = ensemble_predict(models, deletion_images, x3_instance)
    denorm_new_predictions  = [new_prediction * vott_target_test_std + vott_target_test_mean for new_prediction in new_predictions]
    denormalized_H_new_predictions  = [H_station - denorm_new_prediction for denorm_new_prediction in denorm_new_predictions]
    # Calcolo del mse rispetto la predizione originale
    errors = [mean_squared_error(original_prediction, masked_pred) for masked_pred in denormalized_H_new_predictions]

    initial_error = 0.0
    print(f"Initial Prediction with Original instance, prediction: {original_prediction}, error: {initial_error}")

    for nr_superpixel, error in enumerate(errors):
      print(f"Removed SuperPixel: {sorted_per_importance_all_superpixels_index[nr_superpixel]}, new prediction: {denormalized_H_new_predictions[nr_superpixel]}, error: {error}")

    total_errors = [initial_error] + errors # Errore iniziale + errori su tutti i pixel rimossi

    # Plot
    # Nuovo asse X: numero di superpixel inseriti (1, 2, ..., 8)
    x = np.arange(0, len(total_errors))  # Da 0 a 8 inclusi
    #print(x)
    x_for_auc = np.linspace(0, 1, len(total_errors))
    # Calcolo dell'AUC con il nuovo asse x
    auc = calculate_auc(x_for_auc, total_errors)
    print(f"Area under the curve (AUC): {auc}")

    # Normalizzazione dell'asse X tra 0 e 1
    #x = np.linspace(0, 1, len(total_errors))

    # # Creazione del plot
    # plt.figure(figsize=(7,5))
    # plt.plot(x, total_errors, marker='o', linestyle='-', color='red')

    # # Pallini rossi sui punti della curva
    # #plt.scatter(x, total_errors, color='red', zorder=3)

    # # Area sotto la curva
    # plt.fill_between(x, total_errors, color='lightcoral', alpha=0.4)

    # # Testo "Error curve" in alto a sx con font più piccolo
    # plt.legend(['Error curve'], loc='lower right',  bbox_to_anchor=(0.97, 0.02))

    # # Testo AUC leggermente spostato sotto la legenda
    # plt.text(0.941, 0.13, f'AUC: {auc:.2f}',
    #          transform=plt.gca().transAxes,
    #          fontsize=10,
    #          verticalalignment='bottom',
    #          horizontalalignment='right',
    #          bbox=dict(facecolor='white', alpha=0.6, edgecolor='grey'))

    # # Etichette degli assi
    # plt.xlabel('Numbers of pixels removed')
    # plt.ylabel('Mean Squared Error')
    # plt.xticks(x)  # Imposta i tick esattamente sui numeri interi (1, 2, ..., 8)
    # # Titolo del grafico
    # plt.title("Deletion Metric Curve")

    # # Mostra il grafico
    # plt.show()
    return total_errors,auc

def calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients, spatial_superpixels, models=vott_lstm_models_loaded, 
                                                           H_station=390.0, channel_prec=0, std_zero_value=-0.6486319166678826,input_size=(104,5,8,3),T=104,H=5,W=8):
  
  instance    = copy.deepcopy(vottignasco_test_image[nr_instance])
  x3_instance = copy.deepcopy(vottignasco_test_OHE[nr_instance])

  abs_coefficients = np.abs(coefficients)

  saliency_map_i     = np.zeros((H,W))
  saliency_map_i_abs = np.zeros((H,W))
  # Creo la mappa di salienza con i coefficienti nei superpixels 
  for i,superpixel in enumerate(spatial_superpixels):
          saliency_map_i     += coefficients[i] * superpixel
          saliency_map_i_abs += abs_coefficients[i] * superpixel

  # Ranking sui coefficienti in abs
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


"""### ***Experiments***

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

# Canali
channel_prec, channel_tmax, channel_tmin = 0, 1, 2
channels = [channel_prec, channel_tmax, channel_tmin]

# Modelli e dimensioni
models = vott_lstm_models_loaded
T, H, W, C = (104, 5, 8, 3)
input_size_spatial = (H, W)
std_zero_value = -0.6486319166678826

# Parametri SLIC
slic_param = [(4, 20), (4, 25), (7,4), (4, 10), (7, 10), (8,2), (8,15), (9,20)]
#slic_param = [(8,2)]

# Valori alpha e kernel width per LIME
alpha_values = [0.1, 10.0]
#alpha_values = [10.0]
percentile_kernel_width_values = [50, 90]
#percentile_kernel_width_values = [90]

# Lunghezza del test set
len_test_set = len(vottignasco_test_image)

# Loop principale sui parametri SLIC
for nr_setup, slic_p in enumerate(slic_param):
    print(f"############################## Setup #{nr_setup}: {slic_p} ##############################")
    n_s, comp = slic_p
    param_combination = f"ns_{n_s}_comp_{comp}"

    # Dizionario per salvare i risultati
    results = {"lime": {}, "shap": {}}
    print(f"############################## Parameters Combination: {param_combination} ##############################")

    # Loop sulle istanze del test set
    for nr_instance, _ in enumerate(vottignasco_test_image):
        print(f"###################### Explanation for Instance #{nr_instance} ####################################")
        base_start_time_lime_shap = datetime.datetime.now()

        superpixels, zs_primes, perturbed_instances, preds_masked = lime_shap(nr_instance, vottignasco_test_image, vottignasco_test_OHE, channels, models,
                                                                              n_s, comp, input_size_spatial, std_zero_value)
        
        base_end_time_lime_shap = datetime.datetime.now()
        exec_time_base_lime_shap = base_end_time_lime_shap - base_start_time_lime_shap

        nr_coefficients = len(superpixels)
        instance = copy.deepcopy(vottignasco_test_image[nr_instance])

        # Prepara input per il regressore
        X = np.array([z.flatten() for z in zs_primes])  # Maschere come righe
        y = np.array(preds_masked)  # Predizioni corrispondenti

        ################# SHAP #################################
        time_start_shap = datetime.datetime.now()

        # SHAP: calcolo dei pesi e regressione
        M = len(superpixels)
        weights_shap = calculate_weigths_shap(M, zs_primes)
        regressor_shap = Ridge(alpha=0.0)
        regressor_shap.fit(X, y, sample_weight=weights_shap)
        coefficients_shap = regressor_shap.coef_

        time_end_shap = datetime.datetime.now()
        exec_time_shap =  (exec_time_base_lime_shap + (time_end_shap - time_start_shap)).total_seconds()

        param_combination_shap = f"ns_{n_s}_comp_{comp}"

        #Insertion and Deletion
        saliency_map_shap_i, errors_insertion_shap,auc_insertion_shap, errors_deletion_shap,auc_deletion_shap = calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients_shap, superpixels)

        # Inizializza solo una volta il dizionario per SHAP
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
        
        # Salva i coefficienti per ogni istanza
        results["shap"][param_combination_shap]["coefficients"][nr_instance, :] = coefficients_shap
        results["shap"][param_combination_shap]["saliency_maps"][nr_instance, :] = saliency_map_shap_i
        results["shap"][param_combination_shap]["errors_insertion"][nr_instance, :] = errors_insertion_shap
        results["shap"][param_combination_shap]["auc_insertion"][nr_instance, :] = auc_insertion_shap
        results["shap"][param_combination_shap]["errors_deletion"][nr_instance, :] = errors_deletion_shap
        results["shap"][param_combination_shap]["auc_deletion"][nr_instance, :] = auc_deletion_shap
        results["shap"][param_combination_shap]["executions_times"][nr_instance, :] = exec_time_shap
        ########################################## END SHAP ########################################

        ######################################## LIME ############################################
        # LIME: ciclo su iperparametri
        for alpha in alpha_values:
            for kernel_width_p in percentile_kernel_width_values:
                time_start_lime = datetime.datetime.now()

                weights_lime = calculate_weigths_lime(instance, perturbed_instances, percentile_kernel_width=kernel_width_p)
                regressor_lime = Ridge(alpha=alpha)
                regressor_lime.fit(X, y, sample_weight=weights_lime)
                coefficients_lime = regressor_lime.coef_

                time_end_lime = datetime.datetime.now()
                exec_time_lime = (exec_time_base_lime_shap + (time_end_lime - time_start_lime)).total_seconds()

                param_combination_lime = f"ns_{n_s}_comp_{comp}_kw_{kernel_width_p}_alpha_{alpha}"

                # Calcolo Saliency_Map, Insertion/Deletion 
                saliency_map_lime_i, errors_insertion_lime,auc_insertion_lime, errors_deletion_lime,auc_deletion_lime = calculate_saliency_map_insertion_deletion_errors_auc(nr_instance, coefficients_lime, superpixels)

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

                # Salvataggio dei coefficienti
                results["lime"][param_combination_lime]["coefficients"][nr_instance, :] = coefficients_lime
                results["lime"][param_combination_lime]["saliency_maps"][nr_instance, :] = saliency_map_lime_i
                results["lime"][param_combination_lime]["errors_insertion"][nr_instance, :] = errors_insertion_lime
                results["lime"][param_combination_lime]["auc_insertion"][nr_instance, :] = auc_insertion_lime
                results["lime"][param_combination_lime]["errors_deletion"][nr_instance, :] = errors_deletion_lime
                results["lime"][param_combination_lime]["auc_deletion"][nr_instance, :] = auc_deletion_lime
                results["lime"][param_combination_lime]["executions_times"][nr_instance, :] = exec_time_lime

    # Salva risultati una volta per ogni setup
    #path_to_save_results = f"/content/spatial_results_setup_ns_{n_s}_comp_{comp}.pkl"
    #path_to_save_results = os.path.join(work_path, f"Water_Resources/rise-video/XAI/spatial/results/lime_shap_multiplicative_norm_zero/spatial_results_setup_ns_{n_s}_comp_{comp}.pkl")
    path_to_save_results = f"{RESULT_DIR}/lime_shap_spatial_results_setup_ns_{n_s}_comp_{comp}.pkl"

    with open(path_to_save_results, 'wb') as f:
        pickle.dump(results, f)

    print(f"Risultati salvati in {path_to_save_results}")

