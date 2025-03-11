import numpy as np


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