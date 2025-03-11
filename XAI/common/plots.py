

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

    # # Plot per la curva di inserimento
    # max_len_insertion = max(map(len, all_errors_insertion))
    # padded_array_insertion = np.full((len(all_errors_insertion), max_len_insertion), np.nan)
    # for i, row in enumerate(all_errors_insertion):
    #     padded_array_insertion[i, :len(row)] = row  # Riempie solo le parti esistenti
    # mean_errors_for_insertion_vott = np.nanmean(padded_array_insertion, axis=0)
    mean_errors_for_insertion_vott = np.nanmean(all_errors_insertion, axis=0)

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

    # # Plot per la curva di cancellazione
    # max_len_deletion = max(map(len, all_errors_deletion))
    # padded_array_deletion = np.full((len(all_errors_deletion), max_len_deletion), np.nan)
    # for i, row in enumerate(all_errors_deletion):
    #     padded_array_deletion[i, :len(row)] = row  # Riempie solo le parti esistenti
    # mean_errors_for_deletion_vott = np.nanmean(padded_array_deletion, axis=0)
    mean_errors_for_deletion_vott = np.nanmean(all_errors_deletion, axis=0)

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