

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

################################## RISE-Spatial #########################################


def plot_frame(frame, cmap="PuBu", title="", ax=None):
    if ax is None:  # Se non viene passato un asse, crea un nuovo plot
        plt.imshow(frame, origin="lower", cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.show()
    else:  # Usa l'asse specifico (per la griglia)
        im = ax.imshow(frame, origin="lower", cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

import numpy as np
import xarray
import matplotlib.pyplot as plt
import cmasher as cmr

import xarray
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box, LineString, MultiLineString
import cmasher as cmr

# piedmont_shp = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/shapefile_raster/Ambiti_Amministrativi-Province.shp" 

# piedmont_bounds = gpd.read_file(piedmont_shp)
# piedmont_bounds = piedmont_bounds.to_crs('epsg:4326')
# # remove the small enclaved Cuneo area inside Torino province
# piedmont_bounds = piedmont_bounds[:-1]

# # extract bound, useful for plots
# piemonte_long_min, piemonte_lat_min, piemonte_long_max, piemonte_lat_max = piedmont_bounds.total_bounds

# piedmont_bounds.boundary.plot()

# # Catchment shapefile
# catchment = gpd.read_file("/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/shapefile_raster/BAC_01_bacialti.shp") # select GRANA-MAIRA	and VARAITA
# catchment = catchment.to_crs('epsg:4326')

# # Select only the Grana-Maira catchment
# catchment_GM = catchment.loc[catchment.NOME == "GRANA-MAIRA"]
# catchment_GM = catchment_GM.reset_index(drop = True)

# # Retrieve the borders of the catchment from the shapefile
# xmin_clip, ymin_clip, xmax_clip, ymax_clip = catchment_GM.total_bounds
# # Extend the borders to include more pixel on the borders

# increase = 0.05 # Degrees
# #ymin_clip -= increase # not needed
# xmin_clip += increase # "+" for subset for pixel included in the mask
# xmax_clip += increase
# #ymax_clip += increase # not needed

# # Define a box around the Region of Interest (ROI)
# ROI_box = box(xmin_clip, ymin_clip, xmax_clip , ymax_clip)

# ROI_shp = piedmont_bounds.clip(ROI_box)
# ROI_shp.boundary.plot()

# meteo_ds = xarray.open_dataset("/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/shapefile_raster/meteo_data_ARPA_GM_clipped.nc") # reading ERA5 file
# # ARPA Water Table sensors in Cuneo and Torino Provinces
# arpa_wt_sensors = gpd.read_file("/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/shapefile_raster/CN_TO_observed_d_t_c_stations.shp")
# arpa_wt_sensors = arpa_wt_sensors.to_crs('epsg:4326')
# arpa_wt_sensors = arpa_wt_sensors.loc[arpa_wt_sensors.Munic.isin(["Vottignasco"])]

def plot_saliency_map_matteo(saliency_map, current_instance, data_target, meteo_ds, 
                            catchment, piedmont_bounds, cmap='seismic', mode="standard",
                            alpha=0.95, title_prefix='Spatial saliency'):
    """
    Plotta la mappa di salienza normalizzata con un range regolabile.
    
    Parametri:
    - saliency_map: array 2D con i valori di salienza (shape: [lat, lon])
    - current_instance: indice corrente per il titolo
    - data_target: data associata all'istanza
    - meteo_ds: dataset con le coordinate lat e lon
    - catchment: confini del bacino da plottare
    - piedmont_bounds: confini del Piemonte da plottare
    - cmap: colormap da usare (default: 'seismic')
    - mode: "standard" per la colormap normale, "reversed" per invertirla
    - alpha: trasparenza della mappa (default: 0.95)
    - title_prefix: prefisso del titolo (default: 'Spatial saliency')
    """
    
    # Converti in DataArray per compatibilità con xarray
    xarray_saliency = xarray.DataArray(
        data=saliency_map,
        dims=["y", "x"],
        coords=dict(y=("y", meteo_ds.coords["lat"].values),
                    x=("x", meteo_ds.coords["lon"].values))
    )
    
    vmin = np.min(xarray_saliency)
    vmax = np.max(xarray_saliency)

    # Seleziona la colormap (invertita o normale)
    selected_cmap = cmr.get_sub_cmap(cmap, 0.5, 1)
    if mode == "reversed":
        selected_cmap = selected_cmap.reversed()

    # Setup del plot
    fig, ax = plt.subplots()
    catchment.boundary.plot(ax=ax, color="Blue", linewidth=0.1, alpha=1)
    piedmont_bounds.boundary.plot(ax=ax, color='Black', linewidth=0.5, alpha=0.6)
    
    # Plotta la mappa di salienza con la colormap selezionata
    xarray_saliency.plot(ax=ax, cmap=selected_cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    
    # Titolo del grafico
    fig.suptitle(f'{title_prefix} #{current_instance} (target: {data_target[0]})', fontsize=12)
    
    plt.tight_layout()
    plt.show()
        

################################## RISE-Temporal ##########################################

# Funzione per determinare la stagione in base al giorno dell'anno
def get_season(day):
    spring = np.arange(80, 172)
    summer = np.arange(172, 264)
    fall = np.arange(264, 355)

    if day in spring:
        return 'Spring'
    elif day in summer:
        return 'Summer'
    elif day in fall:
        return 'Autumn'
    else:
        return 'Winter'

season_colors = {
    'Winter': '#AEC6CF',  # Azzurro chiaro
    'Spring': '#77DD77',  # Verde menta
    'Summer': '#FFD700',  # Giallo dorato
    'Autumn': '#FF8C00'   # Arancione scuro
}

def plot_saliency_vector(saliency_vector, test_dates, test_images_dates, instance_number, mode="standard", input_size=104):
    dates = pd.to_datetime(test_images_dates[instance_number])
    tm_days = [date.timetuple().tm_yday for date in dates]
    seasons = [get_season(tm_yday) for tm_yday in tm_days]
    colors = [season_colors[season] for season in seasons]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(input_size), saliency_vector, color='black')

    # Evidenzia le stagioni come intervalli
    season_intervals = []
    start = 0
    current_season = seasons[0]

    for j in range(1, input_size):
        if seasons[j] != current_season:
            season_intervals.append((start, j, current_season))
            start = j
            current_season = seasons[j]
    season_intervals.append((start, input_size, current_season))

    if mode == "to_zero":
        max_saliency = min(saliency_vector)
        min_saliency = max(saliency_vector)
    else:
        max_saliency = max(saliency_vector)
        min_saliency = min(saliency_vector)
    #print(max_saliency, min_saliency)
    #text_offset = max_saliency * +0.06  # Aggiunge un po' di spazio sopra il valore massimo

    for start, end, season in season_intervals:
        ax.axvspan(start, end, color=season_colors[season], alpha=0.2)
        ax.text((start + end) / 2, max_saliency, season, color=season_colors[season], fontsize=10, ha='center', va='bottom')

    # Seleziona le date a intervalli regolari per i tick dell'asse x
    month_indices = np.linspace(0, input_size - 1, num=12, dtype=int)
    month_labels = [f"{dates[i].strftime('%b')} {dates[i].year}" for i in month_indices]

    ax.set_xticks(month_indices)
    ax.set_xticklabels(month_labels, rotation=45)

    # Linee tratteggiate per separare i time-step
    for i in range(input_size):
        ax.axvline(x=i, color='grey', linestyle='-', alpha=0.1)

    # Impostare i limiti dell'asse y tra il minimo e massimo della saliency
    ax.set_ylim(min_saliency, max_saliency + (max_saliency - min_saliency) * 0.05)  # Padding solo sopra il max

    
    # Impostare i tick dell'asse y automaticamente in base ai valori della saliency
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))  # Genera tick ragionevoli

    ax.set_title(f"Temporal Saliency Vector for Instance #{instance_number}, Target Date: {test_dates[instance_number][0].astype(str).split('T')[0]}\n(Impact of each week on groundwater level prediction)", pad=26)
    #ax.set_xlabel('Time-step')
    ax.set_ylabel('Saliency score (Relevance of each time-step)')

    fig.subplots_adjust(left=0.1, right=0.9)
    fig.tight_layout()
    plt.show()

from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable 
from matplotlib.collections import LineCollection

# Funzione per determinare la stagione in base al giorno dell'anno
def get_season(day):
    spring = np.arange(80, 172)
    summer = np.arange(172, 264)
    fall = np.arange(264, 355)

    if day in spring:
        return 'Spring'
    elif day in summer:
        return 'Summer'
    elif day in fall:
        return 'Autumn'
    else:
        return 'Winter'

season_colors = {
    'Winter': 'darkslategray',#'#AEC6CF',  # Azzurro chiaro
    'Spring': 'darkolivegreen', #'#77DD77',  # Verde menta
    'Summer': 'olivedrab', #'#FFD700',  # Giallo dorato
    'Autumn': 'cadetblue', #'#FF8C00'   # Arancione scuro
}

def plot_saliency_vector_matteo(saliency_vector, test_dates, test_images_dates, instance_number, ylab, input_size=104, reverse_cbar = "False"):
    
    dates = pd.to_datetime(test_images_dates[instance_number])
    tm_days = [date.timetuple().tm_yday for date in dates]
    seasons = [get_season(tm_yday) for tm_yday in tm_days]
    colors = [season_colors[season] for season in seasons]
    
    #Create Barplot:
    data_color = [(x-saliency_vector.min()) / (saliency_vector.max()-saliency_vector.min()) for x in saliency_vector] #see the difference here
    my_cmap = cmr.get_sub_cmap('seismic', 0.5, 1).reversed() if reverse_cbar else cmr.get_sub_cmap('seismic', 0.5, 1)
    colors_bar = my_cmap(data_color)

    fig, ax = plt.subplots(figsize=(12, 0.75))

    # Evidenzia le stagioni come intervalli
    season_intervals = []
    start = 0
    current_season = seasons[0]

    for j in range(1, input_size):
        if seasons[j] != current_season:
            season_intervals.append((start, j, current_season))
            start = j
            current_season = seasons[j]
    season_intervals.append((start, input_size, current_season))

    # if mode == "to_zero":
    #     max_saliency = min(saliency_vector)
    #     min_saliency = max(saliency_vector)
    # else:
    #     max_saliency = max(saliency_vector)
    #     min_saliency = min(saliency_vector)
    #print(max_saliency, min_saliency)
    #text_offset = max_saliency * +0.06  # Aggiunge un po' di spazio sopra il valore massimo
    
    ax.set_ylim(saliency_vector.min() - np.abs(saliency_vector.min()*0.05), saliency_vector.max() + np.abs(saliency_vector.max()*0.05))
    ax.set_xlim(0,103)

    for start, end, season in season_intervals:
        #ax.axvspan(start, end, edgecolor="black", facecolor = None, ls = "-.", zorder = 10)
        ax.axvline(x=start, color='black', linestyle='-.', lw=1, zorder = 10)
        ax.text((start + end) / 2, plt.gca().get_ylim()[1] + np.abs(plt.gca().get_ylim()[1]*0.075), season, color=season_colors[season], fontsize=10, ha='center', va='bottom', zorder = 15)
    ax.axvline(x=end, color='black', linestyle='-.', lw=1, zorder = 10)
        
    
    #ax.bar(np.arange(input_size), saliency_vector, edgecolor = "black", lw = 0.5, facecolor = colors_bar)
    ax.plot(np.arange(input_size), saliency_vector, color = "black", lw = 1.5, zorder = 8, marker = "D", markerfacecolor = "None", markersize = 2)
    #ax.scatter(np.arange(input_size), saliency_vector, c = saliency_vector, cmap = my_cmap, zorder = 9)
    #lwidths=(saliency_vector - saliency_vector.min())*7/(saliency_vector.max() - saliency_vector.min())
    
    # points = np.array([np.arange(input_size), saliency_vector]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # ax.vlines(np.arange(input_size), ymin = plt.gca().get_ylim()[0], ymax = plt.gca().get_ylim()[1],
    #           color = colors_bar, zorder = 7, lw = 7)
    
    ax.bar(np.arange(input_size),
                      bottom= plt.gca().get_ylim()[0],
                      height = abs(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]),
                      width=7,
                      align='edge', color = colors_bar,
                      linewidth = 0.4,
                      ls = "--",
                      edgecolor = "lightgrey",
                      zorder = 7)
    
        # s = np.ones(104) * min_saliency
        # segments=np.array(list(zip(zip(np.arange(input_size),np.arange(input_size)[1:]),zip(s,s[1:])))).transpose((0,2,1))
        # lc = LineCollection(segments, lw = 150, color = colors_bar) #linewidths=lwidths,
        # ax.add_collection(lc)

    # Seleziona le date a intervalli regolari per i tick dell'asse x
    month_indices = np.linspace(0, input_size - 1, num=12, dtype=int)
    month_labels = [f"{dates[i].strftime('%b')} {dates[i].year}" for i in month_indices]

    ax.set_xticks(month_indices)
    ax.set_xticklabels(month_labels, rotation=45)

    # Linee tratteggiate per separare i time-step
    # for i in range(input_size):
    #     ax.axvline(x=i, color='grey', linestyle='--', alpha=0.25, zorder = 8)

    # Impostare i limiti dell'asse y tra il minimo e massimo della saliency
    #ax.set_ylim(min_saliency, max_saliency + (max_saliency - min_saliency) * 0.1)  # Padding solo sopra il max

    
    # Impostare i tick dell'asse y automaticamente in base ai valori della saliency
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both',  nbins=6))  # Genera tick ragionevoli

    ax.set_title(f"Temporal Saliency Vector for Instance #{instance_number}, Target Date: {test_dates[instance_number][0].astype(str).split('T')[0]}\n(Impact of each week on groundwater level prediction)", pad=26)
    #ax.set_xlabel('Time-step')
    ax.set_ylabel(ylab)

    fig.subplots_adjust(left=0.1, right=0.9)
    fig.tight_layout()
    plt.show()

############################# RISE-Spatio_Temporal ###########################################

# Calcolo del valore medio di Salinecy per stagione per ogni istanza di Vottignasco

vottignasco_test_images_dates = np.load("/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_image_sequences_dates.npy")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy




def get_index_range_for_season(seasons):
  indices_range_season = []  # sarà istanziata con triplette (start_index, end_index, season) per ogni stagione

  season_prev = seasons[0]  # Inizializzo con la prima stagione al time-step 0
  start_index = 0

  for i, season in enumerate(seasons):
    if (season != season_prev):
      #print("Ok è cambiata la stagione al time-step", i)
      indices_range_season.append((start_index, i-1, season_prev))
      season_prev = season
      start_index = i

    # Caso in cui negli ultimi time step ho una stagione differente!
    if (i==103):
      last_triple = indices_range_season[-1:]
      if (last_triple[0][2] != season_prev):
        indices_range_season.append((last_triple[0][1] + 1, 103, seasons[103]))

  return indices_range_season

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

def plot_sv_mean_per_season(sv_nr_instance, nr_instance, test_dates, test_images_dates, cmap="PuBu"):
  
  date_target = test_dates[nr_instance].astype('datetime64[D]')

  x1_i_dates = copy.deepcopy(test_images_dates[nr_instance])

  dates = pd.to_datetime(x1_i_dates)

  tm_days = [date.timetuple().tm_yday for date in dates]
  seasons = [get_season(tm_yday) for tm_yday in tm_days]

  # Ottieni le triplette (indice_iniziale, indice_finale, stagione)
  indices_range_season = get_index_range_for_season(seasons)

  # Nel caso di 9 stagioni creo un griglia 3x4, altrimenti 2x4
  if len(indices_range_season) == 9:
    # Imposta i plot in una griglia 3x4
    fig, axes = plt.subplots(3, 4, figsize=(11, 10))
  else:
    fig, axes = plt.subplots(2, 4, figsize=(11, 10))
  axes = axes.flatten()  # Converti la matrice di subplot in una lista

  # Trova i valori minimo e massimo per la colorbar
  vmin = np.min(sv_nr_instance)
  vmax = np.max(sv_nr_instance)

  for i, (index_start, index_end, season) in enumerate(indices_range_season):
      if index_start == index_end:
          sv_mean_season = sv_nr_instance[index_start]
      else:
          sv_mean_season = np.mean(sv_nr_instance[index_start:index_end], axis=0)

      im = axes[i].imshow(sv_mean_season, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
      year = x1_i_dates[index_end].astype('datetime64[Y]').astype(int) + 1970
      axes[i].set_title(f'Mean Saliency-Video - {season}, {year}', fontdict={'fontsize': 8})
      x_ticks = np.arange(0, 8, step=1)  # crea tick ogni 1
      axes[i].set_xticks(x_ticks)

  # Disattiva gli assi vuoti
  for j in range(i + 1, len(axes)):
      axes[j].axis('off')

  # Riduci lo spazio bianco tra i subplot
  plt.tight_layout()

  if len(indices_range_season) == 9:
    # Aggiusta i margini e riduci ulteriormente hspace e wspace per avvicinare i subplot
    fig.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.40, hspace=0.12, wspace=0.12)
    # Aggiungi la colorbar a destra dei subplot
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.05, use_gridspec=True, aspect=25)
  else:
    fig.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.60, hspace=0.15, wspace=0.15)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.05, use_gridspec=True, aspect=16)

  cbar.set_label('Saliency', fontsize=11)

  # Titolo principale
  fig.suptitle(f'Mean Saliency-Video per Season on instance no. {nr_instance}, Date: {date_target}', fontsize=16)

  # Salva la figura con dpi=400
  #plt.savefig(f'/content/sv_per_season_vott_norm_{nr_instance}.png', dpi=400, bbox_inches='tight')
  #plt.savefig(f'./MyDrive/Water_Resources/results/spatial-temporal/sv_mean_per_season/sv_per_season_vott_norm_{nr_instance}.png', dpi=400, bbox_inches='tight')

  plt.show()
  plt.close(fig)

################## Evalutation Metrics: Insertion/Deletion ###################################

def calculate_auc(x, y):
    """
    Calcola l'area sotto la curva (AUC) utilizzando il metodo del trapezio.

    :param x: Valori dell'asse x (frazione dei pixel/frame inseriti).
    :param y: Valori dell'asse y (errori calcolati).
    :return: Area sotto la curva.
    """
    return np.trapz(y, x)

def plot_combined_curves(all_errors_insertion, all_errors_deletion, title="", save=False, save_path=""):
    # Creazione della figura e dei due subplot (1 riga, 2 colonne)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    # Plot per la curva di inserimento
    max_len_insertion = max(map(len, all_errors_insertion))
    padded_array_insertion = np.full((len(all_errors_insertion), max_len_insertion), np.nan)
    for i, row in enumerate(all_errors_insertion):
        padded_array_insertion[i, :len(row)] = row  # Riempie solo le parti esistenti
    #mean_errors_for_insertion_vott = np.nanmean(all_errors_insertion, axis=0)
    mean_errors_for_insertion_vott = np.nanmean(padded_array_insertion, axis=0)

    #x_insertion = np.arange(0, mean_errors_for_insertion_vott.shape[0])
    x_insertion = np.linspace(0, 1, mean_errors_for_insertion_vott.shape[0])
    auc_insertion = calculate_auc(x_insertion, mean_errors_for_insertion_vott)
    auc_text_insertion = f'AUC = {auc_insertion:.2f}'
    axs[0].plot(x_insertion, mean_errors_for_insertion_vott, label=f'Error Curve, {auc_text_insertion}')
    #axs[0].scatter(x_insertion, mean_errors_for_insertion_vott, color='blue', zorder=3)


    axs[0].fill_between(x_insertion, mean_errors_for_insertion_vott, color='skyblue', alpha=0.4)
    axs[0].set_xlabel('Fraction of pixels inserted')
    axs[0].set_ylabel('Mean Squared Error')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].set_title('Mean Insertion Metric Curve')
    axs[0].legend()

    # Plot per la curva di cancellazione
    max_len_deletion = max(map(len, all_errors_deletion))
    padded_array_deletion = np.full((len(all_errors_deletion), max_len_deletion), np.nan)
    for i, row in enumerate(all_errors_deletion):
        padded_array_deletion[i, :len(row)] = row  # Riempie solo le parti esistenti
    mean_errors_for_deletion_vott = np.nanmean(padded_array_deletion, axis=0)
    #mean_errors_for_deletion_vott = np.nanmean(all_errors_deletion, axis=0)

    #x_deletion = np.arange(0, mean_errors_for_deletion_vott.shape[0])
    x_deletion = np.linspace(0, 1, mean_errors_for_deletion_vott.shape[0])
    auc_deletion = calculate_auc(x_deletion, mean_errors_for_deletion_vott)
    auc_text_deletion = f'AUC = {auc_deletion:.2f}'

    axs[1].plot(x_deletion, mean_errors_for_deletion_vott, label=f'Error Curve, {auc_text_deletion}')
    #axs[1].scatter(x_deletion, mean_errors_for_deletion_vott, color='red', zorder=3)
    axs[1].fill_between(x_deletion, mean_errors_for_deletion_vott, color='lightcoral', alpha=0.4)
    axs[1].set_xlabel('Fraction of pixels removed')
    axs[1].set_ylabel('Mean Squared Error')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_title('Deletion Mean Metric Curve')
    axs[1].legend()

    # Aggiungi il titolo globale
    plt.suptitle(title, fontsize=16)
    # Mostra i plot
    plt.tight_layout()

    # Salva la figura con DPI 400
    if save == True:
      plt.tight_layout()
      plt.subplots_adjust(top=0.85)  # Per evitare che il titolo si sovrapponga ai grafici
      plt.savefig(save_path, dpi=400)
    else:
      plt.show()

    return auc_insertion, auc_deletion

def plot_combined_curves_with_errors(all_errors_insertion, all_errors_deletion, title="", save=False, save_path=""):
    # Creazione della figura e dei due subplot (1 riga, 2 colonne)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    # Plot per la curva di inserimento
    max_len_insertion = max(map(len, all_errors_insertion))
    padded_array_insertion = np.full((len(all_errors_insertion), max_len_insertion), np.nan)
    for i, row in enumerate(all_errors_insertion):
        padded_array_insertion[i, :len(row)] = row  # Riempie solo le parti esistenti
    mean_errors_for_insertion_vott = np.nanmean(padded_array_insertion, axis=0)
    #mean_errors_for_insertion_vott = np.nanmean(all_errors_insertion, axis=0)

    #x_insertion = np.arange(0, mean_errors_for_insertion_vott.shape[0])
    x_insertion = np.linspace(0, 1, mean_errors_for_insertion_vott.shape[0])
    auc_insertion = calculate_auc(x_insertion, mean_errors_for_insertion_vott)
    auc_text_insertion = f'AUC = {auc_insertion:.2f}'
    axs[0].plot(x_insertion, mean_errors_for_insertion_vott, label=f'Error Curve, {auc_text_insertion}')
    #axs[0].scatter(x_insertion, mean_errors_for_insertion_vott, color='blue', zorder=3)


    axs[0].fill_between(x_insertion, mean_errors_for_insertion_vott, color='skyblue', alpha=0.4)
    axs[0].set_xlabel('Fraction of pixels inserted')
    axs[0].set_ylabel('Mean Squared Error')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].set_title('Mean Insertion Metric Curve')
    axs[0].legend()

    # Plot per la curva di cancellazione
    max_len_deletion = max(map(len, all_errors_deletion))
    padded_array_deletion = np.full((len(all_errors_deletion), max_len_deletion), np.nan)
    for i, row in enumerate(all_errors_deletion):
        padded_array_deletion[i, :len(row)] = row  # Riempie solo le parti esistenti
    mean_errors_for_deletion_vott = np.nanmean(padded_array_deletion, axis=0)
    #mean_errors_for_deletion_vott = np.nanmean(all_errors_deletion, axis=0)

    #x_deletion = np.arange(0, mean_errors_for_deletion_vott.shape[0])
    x_deletion = np.linspace(0, 1, mean_errors_for_deletion_vott.shape[0])
    auc_deletion = calculate_auc(x_deletion, mean_errors_for_deletion_vott)
    auc_text_deletion = f'AUC = {auc_deletion:.2f}'

    axs[1].plot(x_deletion, mean_errors_for_deletion_vott, label=f'Error Curve, {auc_text_deletion}')
    #axs[1].scatter(x_deletion, mean_errors_for_deletion_vott, color='red', zorder=3)
    axs[1].fill_between(x_deletion, mean_errors_for_deletion_vott, color='lightcoral', alpha=0.4)
    axs[1].set_xlabel('Fraction of pixels removed')
    axs[1].set_ylabel('Mean Squared Error')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_title('Deletion Mean Metric Curve')
    axs[1].legend()

    # Aggiungi il titolo globale
    plt.suptitle(title, fontsize=16)
    # Mostra i plot
    plt.tight_layout()

    # Salva la figura con DPI 400
    if save == True:
      plt.tight_layout()
      plt.subplots_adjust(top=0.85)  # Per evitare che il titolo si sovrapponga ai grafici
      plt.savefig(save_path, dpi=400)
    else:
      plt.show()

    return mean_errors_for_insertion_vott,mean_errors_for_deletion_vott,auc_insertion, auc_deletion

def plot_auc_comparison(all_param_auc_insertion, all_param_auc_deletion):
    """
    Plotta due scatter plot affiancati (1x2) per confrontare le AUC delle metriche insertion e deletion.

    Args:
    - all_param_auc_insertion (list): Lista di tuple/liste [AUC, configurazione] per insertion.
    - all_param_auc_deletion (list): Lista di tuple/liste [AUC, configurazione] per deletion.
    """

    # Separare le AUC e le configurazioni per insertion
    auc_values_insertion = [item[0] for item in all_param_auc_insertion]
    configurations_insertion = [item[1] for item in all_param_auc_insertion]

    # Separare le AUC e le configurazioni per deletion
    auc_values_deletion = [item[0] for item in all_param_auc_deletion]
    configurations_deletion = [item[1] for item in all_param_auc_deletion]

    # Impostare la figura con due subplot affiancati (1x2)
    plt.figure(figsize=(15, 6))

    # Primo subplot: Insertion
    plt.subplot(1, 2, 1)
    plt.scatter(configurations_insertion, auc_values_insertion, color='blue')
    plt.title('AUC Mean Insertion Values')
    plt.xlabel('Configurations')
    plt.ylabel('AUC')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    # Secondo subplot: Deletion
    plt.subplot(1, 2, 2)
    plt.scatter(configurations_deletion, auc_values_deletion, color='red')
    plt.title('AUC Mean Deletion Values')
    plt.xlabel('Configurations')
    plt.ylabel('AUC')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    # Ottimizzare il layout
    plt.tight_layout()
    plt.show()

def plot_multiple_auc_comparisons(data, labels=None, title='AUC Comparison for Different Parameters'):
    """
    Plotta uno scatter plot comparativo delle AUC per più combinazioni di parametri.

    Args:
    - data (list of lists): Ogni sotto-lista ha coppie [AUC, configurazione]
    - labels (list): Etichette per ogni serie di dati (opzionale)
    - title (str): Titolo del grafico
    """

    plt.figure(figsize=(10, 5))

    # Se non ci sono labels, crea etichette generiche (Set 1, Set 2, ...)
    if labels is None:
        labels = [f'Set {i+1}' for i in range(len(data))]

    # Iterare sui dati e plottare ciascun set
    for i, auc_param_list in enumerate(data):
        auc_values = [item[0] for item in auc_param_list]
        configurations = [item[1] for item in auc_param_list]
        plt.scatter(configurations, auc_values, label=labels[i])

    # Personalizzazioni del plot
    plt.title(title)
    plt.xlabel('Configurations')
    plt.ylabel('AUC')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Parameter Sets")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#######################################################################################################

############### Stability ################################

import matplotlib.pyplot as plt
import numpy as np

def plot_curves_with_stats(curves, title='Curva Media e Deviazione Standard', color='blue', plot_type="insertion", fig_name="ins_del_fig"):
    """
    Plotta più curve con la curva media tratteggiata e la banda della deviazione standard.
    
    Args:
        curves (list or np.array): Matrice (n_curves, n_points) contenente le curve.
        title (str): Titolo del grafico.
        color (str): Colore principale per la curva media e la banda della dev. standard.
    """
    # Converti le curve in array numpy
    curves = np.array(curves)
    
    # Asse x basato sul numero di punti delle curve
    x = np.linspace(0, curves.shape[1] - 1, curves.shape[1]) / curves.shape[1]
    
    # Calcolo della media e della deviazione standard punto per punto
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    
    # Plot delle curve individuali
    plt.figure(figsize=(10, 6))
    for curve in curves:
        plt.plot(x, curve, color='skyblue', alpha=0.5)
    
    auc = calculate_auc(x, mean_curve)
    # Aggiunta della curva media tratteggiata
    plt.plot(x, mean_curve, color=color, linestyle='--', linewidth=2, label=f'ST-SHAP AUC:{auc:.3f}')
    
    # Aggiunta della banda della deviazione standard
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)
    
    # Testo con la deviazione standard media
    # plt.text(x[len(x) // 2], np.max(mean_curve), f'Dev. Std. Media: {np.mean(std_curve):.3f}', fontsize=10)
    
    # Personalizzazione del grafico
    plt.title(title)

    if plot_type=="insertion":
        plt.xlabel('% pixels inserted')
    elif plot_type=="deletion":
        plt.xlabel('% pixels deleted')
    
    plt.ylabel('Mean Squared Error')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if fig_name:
        plt.savefig(f'/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/scripts/notebooks/paper/{fig_name}.png', dpi=400, bbox_inches='tight')

    
    # Mostra il plot
    plt.show()
    plt.close()


def plot_ins_del_curves_paper(curves, title='Curva Media e Deviazione Standard', color='blue', plot_type="insertion", fig_name=None, std=True, method="RISE"):
    """
    Plotta più curve con la curva media tratteggiata e la banda della deviazione standard.
    
    Args:
        curves (list or np.array): Matrice (n_curves, n_points) contenente le curve.
        title (str): Titolo del grafico.
        color (str): Colore principale per la curva media e la banda della dev. standard.
    """
    # Converti le curve in array numpy
    (s_curves, t_curves, st_curves) = curves
    s_curves_np = np.array(s_curves)
    t_curves_np = np.array(t_curves)
    st_curves_np = np.array(st_curves)
    
    # Asse x basato sul numero di punti delle curve
    x_s = np.linspace(0, s_curves_np.shape[1] - 1, s_curves_np.shape[1]) / (s_curves_np.shape[1]-1)
    x_t = np.linspace(0, t_curves_np.shape[1] - 1, t_curves_np.shape[1]) / (t_curves_np.shape[1]-1)
    x_st = np.linspace(0, st_curves_np.shape[1] - 1, st_curves_np.shape[1]) / (st_curves_np.shape[1]-1)
    
    # Calcolo della media e della deviazione standard punto per punto
    s_mean_curve = np.mean(s_curves_np, axis=0)
    s_std_curve = np.std(s_curves_np, axis=0)
    t_mean_curve = np.mean(t_curves_np, axis=0)
    t_std_curve = np.std(t_curves_np, axis=0)
    st_mean_curve = np.mean(st_curves_np, axis=0)
    st_std_curve = np.std(st_curves_np, axis=0)
    
    # Plot delle curve individuali
    plt.figure(figsize=(10, 6))

    s_rise_auc = calculate_auc(x_s, s_mean_curve)
    t_rise_auc = calculate_auc(x_t, t_mean_curve)
    st_rise_auc = calculate_auc(x_st, st_mean_curve)
    
    # Aggiunta della curva media tratteggiata
    plt.plot(x_s, s_mean_curve, color="skyblue", linestyle='--', linewidth=2, label=f'S-{method} AUC:{s_rise_auc:.3f}')
    plt.plot(x_t, t_mean_curve, color="green", linestyle='--', linewidth=2, label=f'T-{method} AUC:{t_rise_auc:.3f}')
    plt.plot(x_st, st_mean_curve, color="red", linestyle='--', linewidth=2, label=f'ST-{method} AUC:{st_rise_auc:.3f}')
    
    # Aggiunta della banda della deviazione standard
    if std:
        plt.fill_between(x_s, s_mean_curve - s_std_curve, s_mean_curve + s_std_curve, color="skyblue", alpha=0.2)
        plt.fill_between(x_t, t_mean_curve - t_std_curve, t_mean_curve + t_std_curve, color="green", alpha=0.2)
        plt.fill_between(x_st, st_mean_curve - st_std_curve, st_mean_curve + st_std_curve, color="red", alpha=0.2)
    
    
    # Personalizzazione del grafico
    plt.title(title)

    if plot_type=="insertion":
        plt.xlabel('% pixels inserted')
    elif plot_type=="deletion":
        plt.xlabel('% pixels deleted')
    
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/scripts/notebooks/paper/{fig_name}.png', dpi=400, bbox_inches='tight')

    
    # Mostra il plot
    plt.show()
    plt.close()