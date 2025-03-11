import numpy as np
import cv2
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator


def generate_masks_1d(N, input_size, seed=42, **kwargs):
    """
    Parametri:
    - input_size: è il nr di time-step -> scalare
    """
    l = kwargs.get("l", 3)  # La lunghezza della small_mask
    p1 = kwargs.get("p1", 0.5)  # Probabilità di attivazione della maschera

    np.random.seed(seed)

    # Genera una maschera 1D casuale (lunghezza = small_mask_length)
    grid = np.random.rand(N, l) < p1
    grid = grid.astype('float32')  # Trasforma in formato float32

    # Crea una struttura per le maschere finali
    masks = np.empty((N, input_size))  # Maschere finali di dimensione (N, H)

    for i in tqdm(range(N), desc='Generating masks'):
        # Calcola i punti di interpolazione
        x = np.linspace(0, l - 1, l)  # Indici della maschera piccola
        new_x = np.linspace(0, l - 1, input_size)  # Nuovi punti per la dimensione H

        # Interpolazione 1D
        interpolated_mask = np.interp(new_x, x, grid[i])  # Interpola la maschera

        # Applica la maschera interpolata alla maschera finale
        masks[i, :] = interpolated_mask

    # Filtra le maschere che sono tutte 0.0
    #masks = masks[~(masks == 0).all(axis=1)]  # Filtra lungo l'asse della dimensione 1 (H)

    return masks

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

def generate_masks_3d(N, input_size, seed=42, **kwargs):
    """
    Parametri:
    - input_size: (t, h, w) -> dimensione finale della maschera 3D (time, Height, Width)
    """
    T, H, W = input_size
    l = kwargs.get("l", 8)   # Dimensione della small mask per il tempo
    h = kwargs.get("h", 2)
    w = kwargs.get("w", 4)
    p1 = kwargs.get("p1", 0.5)  # Probabilità di attivazione

    np.random.seed(seed)

    # Genera una maschera 3D casuale di dimensione (l, h,w)
    grid = np.random.rand(N, l, h, w) < p1
    grid = grid.astype('float32')

    # Struttura per le maschere finali di dimensione (N, T,H,W)
    masks = np.empty((N, T, H, W))

    # Coordinate per l'interpolazione spaziale
    grid_x = np.linspace(0, h - 1, h)
    grid_y = np.linspace(0, w - 1, w)
    grid_t = np.linspace(0, l - 1, l)

    for i in tqdm(range(N), desc='Generating masks'):
      # Crea un interpolatore per la maschera corrente
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