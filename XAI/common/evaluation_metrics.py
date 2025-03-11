import numpy as np
import copy 
from sklearn.metrics import mean_squared_error

import black_box

# Utils

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


""" RISE-Spatial """

# Insertion 

def update_instance_with_pixels_spatial(current_instance, original_instance, x,y):
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
        I_prime = update_instance_with_pixels_spatial(I_prime, original_instance, x,y)
        insertion_images.append(I_prime)

    insertion_images = [img.astype(np.float32) for img in insertion_images]
    # Calcolo le predizioni sulle istanze a cui ho aggiunto i pixel in maniera graduale
    new_predictions = black_box.ensemble_predict(model, insertion_images, x3_instance)
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