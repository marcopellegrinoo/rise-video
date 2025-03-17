import numpy as np
import os
import pickle

# IMPORTO I DATI PER VOTTIGNASCO

# Ottieni il percorso effettivo da una variabile d'ambiente
work_path = os.environ['WORK']  # Ottieni il valore della variabile d'ambiente WORK
v_test_OHE_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_month_OHE.npy"
v_test_image_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_normalized_image_sequences.npy"
v_test_target_dates_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_target_dates.npy"
v_test_images_dates = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/Vottignasco_00425010001_test_image_sequences_dates.npy"
v_test_normalization_factors_std_path  = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/normalization_factors/Vottignasco_00425010001_training_target_std.npy"
v_test_normalization_factors_mean_path = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/normalization_factors/Vottignasco_00425010001_training_target_mean.npy"

# Carica l'array numpy dai file
vottignasco_test_OHE         = np.load(v_test_OHE_path)
vottignasco_test_image       = np.load(v_test_image_path)
vottignasco_test_dates       = np.load(v_test_target_dates_path)
vottignasco_test_image_dates = np.load(v_test_images_dates)
vott_target_test_std         = np.load(v_test_normalization_factors_std_path) 
vott_target_test_mean        = np.load(v_test_normalization_factors_mean_path)


print(len(vottignasco_test_dates))
print(len(vottignasco_test_image))
print(len(vottignasco_test_OHE))

import xarray
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box, LineString, MultiLineString
import cmasher as cmr

piedmont_shp = "/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/shapefile_raster/Ambiti_Amministrativi-Province.shp" 

piedmont_bounds = gpd.read_file(piedmont_shp)
piedmont_bounds = piedmont_bounds.to_crs('epsg:4326')
# remove the small enclaved Cuneo area inside Torino province
piedmont_bounds = piedmont_bounds[:-1]

# extract bound, useful for plots
piemonte_long_min, piemonte_lat_min, piemonte_long_max, piemonte_lat_max = piedmont_bounds.total_bounds

piedmont_bounds.boundary.plot()

# Catchment shapefile
catchment = gpd.read_file("/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/shapefile_raster/BAC_01_bacialti.shp") # select GRANA-MAIRA	and VARAITA
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

# Define a box around the Region of Interest (ROI)
ROI_box = box(xmin_clip, ymin_clip, xmax_clip , ymax_clip)

ROI_shp = piedmont_bounds.clip(ROI_box)
ROI_shp.boundary.plot()

meteo_ds = xarray.open_dataset("/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/shapefile_raster/meteo_data_ARPA_GM_clipped.nc") # reading ERA5 file
# ARPA Water Table sensors in Cuneo and Torino Provinces
arpa_wt_sensors = gpd.read_file("/leonardo_work/try25_pellegrino/Water_Resources/rise-video/data/Vottignasco/shapefile_raster/CN_TO_observed_d_t_c_stations.shp")
arpa_wt_sensors = arpa_wt_sensors.to_crs('epsg:4326')
arpa_wt_sensors = arpa_wt_sensors.loc[arpa_wt_sensors.Munic.isin(["Vottignasco"])]

import copy

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

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm, colors

import pandas as pd
import numpy as np

import geopandas as gpd
import cmasher as cmr


def subset_video_percentile(video, perc, exclude_higher = True):
    
    subsetted_video = copy.deepcopy(video)
    
    percenile_video = np.percentile(subsetted_video, perc)
    
    if exclude_higher:
            subsetted_video[subsetted_video > percenile_video] = np.nan
            
    else: 
            subsetted_video[subsetted_video < percenile_video] = np.nan
    
    return subsetted_video

def normalize_min_max(x, min, max):
    return (x - min) / (max - min)

def denormalize_min_max(x_norm, min, max):
    return ((max - min) * x_norm) + min

def plot_saliency_video_3d(video, perc, test_images_dates, instance_number,
                           elev = None, azim = None, roll = None,
                           high_worst = True, title_ext = "S4",
                           shapefile = None, ROI_box = None, clip_bounds = None,
                           target_shapefile = None,
                           save=False, save_path=""):
            
    subsetted_silency = subset_video_percentile(video, perc, exclude_higher=high_worst)
    
    my_cmap = cmr.get_sub_cmap('seismic', 0.55, 1) #.reversed()
    if high_worst:
        my_cmap = my_cmap.reversed()
    
    norm_silency = (subsetted_silency - np.nanmin(subsetted_silency))/(np.nanmax(subsetted_silency) - np.nanmin(subsetted_silency))
    colors = my_cmap(norm_silency)
    colors = np.where(np.isnan(np.repeat(np.expand_dims(subsetted_silency, -1), 4, -1)), np.full_like(colors, np.nan), colors)

    fig = plt.figure(figsize = (25,8))
    ax = fig.add_subplot(111, projection='3d')
    
    nx, nz = subsetted_silency[0,:,:].shape
    xi, zi = np.mgrid[0:nx+1, 0:nz+1]
    
    # Marginal Plots
    yi_spatial_mean = np.full_like(zi, 107)
    saliency_spatial_mean = np.nanmean(subsetted_silency, axis = 0)
    
    saliency_temporal_mean = np.nanmean(subsetted_silency, axis = (1,2))
    saliency_temporal_mean = np.repeat(np.expand_dims(saliency_temporal_mean, -1), subsetted_silency.shape[1], -1)
    
    ny_temp, nz_temp= saliency_temporal_mean.shape
    y_temp, z_temp = np.mgrid[0:ny_temp, 0:nz_temp]
    xi_temp = np.full_like(y_temp, -2)
    
    my_cmap_marginal =  cmr.get_sub_cmap('Greys', 0, 0.85)
    if high_worst:
        my_cmap_marginal = my_cmap_marginal.reversed()
    colors_spatial_mean = my_cmap_marginal((saliency_spatial_mean - np.nanmin(subsetted_silency))/(np.nanmax(subsetted_silency) - np.nanmin(subsetted_silency)))
    colors_temporal_mean = my_cmap_marginal((saliency_temporal_mean - np.nanmin(subsetted_silency))/(np.nanmax(subsetted_silency) - np.nanmin(subsetted_silency)))
    
    colors_spatial_mean = np.where(np.isnan(np.repeat(np.expand_dims(saliency_spatial_mean, -1), 4, -1)), np.full_like(colors_spatial_mean, np.nan), colors_spatial_mean)
    colors_temporal_mean = np.where(np.isnan(np.repeat(np.expand_dims(saliency_temporal_mean, -1), 4, -1)), np.full_like(colors_temporal_mean, np.nan), colors_temporal_mean)
    
    ax.plot_surface(zi, yi_spatial_mean, xi, facecolors=colors_spatial_mean, rstride=1, cstride=1, 
                    vmin = np.nanmin(subsetted_silency), vmax = np.nanmax(subsetted_silency), zorder = 0.3)
    
    ax.plot_surface(xi_temp, y_temp, z_temp, facecolors=colors_temporal_mean, rstride=1, cstride=1,  lw =4,
                    vmin = np.nanmin(subsetted_silency), vmax = np.nanmax(subsetted_silency), zorder = 0.3)
    
    # Saliency patches

    for i in range(subsetted_silency.shape[0]):
        yi = np.full_like(zi, i)
        frame_color = colors[i]
        ax.plot_surface(zi, yi, xi, rstride=1, cstride=1, facecolors=frame_color, lw = 4, zorder = 0.5)
        ax.set_zticks(np.arange(nx))
        ax.view_init(elev, azim, roll)
        
    # plot colorbar
    
    m = cm.ScalarMappable(cmap=my_cmap)
    m.set_array(np.unique(subsetted_silency[~np.isnan(subsetted_silency)].flatten()))

    col = plt.colorbar(m, ax = ax, label = title_ext)

    # Shapefile
    if shapefile is not None:
        y_offset = 107
        geom = shapefile.clip(ROI_box).boundary
        for el in geom:
            if isinstance(el, (LineString, MultiLineString)):
                    for line in [el] if isinstance(el, LineString) else el.geoms:
                        coords = np.array(line.coords)
                        x_normalized = normalize_min_max(coords[:, 0], clip_bounds[0], clip_bounds[1])
                        z_normalized = normalize_min_max(coords[:, 1], clip_bounds[2], clip_bounds[3])
                        ax.plot(x_normalized*8, y_offset, z_normalized*5, color='black', alpha=0.7, zorder = 5)
                        

    # Target point
    if target_shapefile is not None:
        point_cords = np.array([[target_shapefile.geometry.x.values[0],
                                target_shapefile.geometry.y.values[0]]])
        
        x_point_norm = normalize_min_max(point_cords[:, 0], clip_bounds[0], clip_bounds[1])
        z_point_norm = normalize_min_max(point_cords[:, 1], clip_bounds[2], clip_bounds[3])
        
        ax.scatter( x_point_norm*8, y_offset, z_point_norm*5, zorder = 10, color = "tab:blue", s = 20)
        
    # Customize gridlines:
    # Change the style of the x, y, and z grid lines
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.2)  # RGBA color (gray, 60% opacity)
        axis._axinfo["grid"]["linestyle"] = "--"  # Dashed gridlines
        axis._axinfo["grid"]["linewidth"] = 0.75   # Set gridline thickness
        #axis.set_pane_color((0/255, 0/255, 0/255, 0.75))
    
    # Seasons
    dates = pd.to_datetime(test_images_dates[instance_number])
    tm_days = [date.timetuple().tm_yday for date in dates]
    seasons = [get_season(tm_yday) for tm_yday in tm_days]
    
    # Evidenzia le stagioni come intervalli
    season_intervals = []
    start = 0
    current_season = seasons[0]

    for j in range(1, 104):
        if seasons[j] != current_season:
            season_intervals.append((start, j, current_season))
            start = j
            current_season = seasons[j]
    season_intervals.append((start, 103, current_season))
        
        
    #ax.set_xlim(0,103)
    
    z = np.repeat(0, 100) 
    x = np.linspace(start=-2, stop=9, num=100) 
    
    for start, end, season in season_intervals:
            #ax.axvspan(start, end, edgecolor="black", facecolor = None, ls = "-.", zorder = 10)
            y = np.repeat(start, 100)
            #ax.axvline(x, start, z, color='black', linestyle='-.', lw=1, zorder = 10)
            ax.plot(x, y, z, c="gray", linestyle='-.', lw = 0.75)
            # ax.text((start + end) / 2, plt.gca().get_ylim()[1] + np.abs(plt.gca().get_ylim()[1]*0.075), season, color=season_colors[season], fontsize=10, ha='center', va='bottom', zorder = 15)
            # ax.axvline(x=end, color='black', linestyle='-.', lw=1, zorder = 10)
            
            ax.text(9, (start + end) / 2, 0, zdir = "x", color=season_colors[season], s=season, ha='right', va='baseline') #ha='left', va='center_baseline'
    
    # Axes settings     
    ax.axes.set_xlim3d(left=-2, right=9) 
    ax.axes.set_ylim3d(bottom=0, top=107) 
    ax.axes.set_zlim3d(bottom=0, top=5) 
    ax.set_xticks(np.arange(0,7, 2), labels = denormalize_min_max(np.arange(0,7, 2)/8, clip_bounds[0], clip_bounds[1]).round(2))
    ax.set_xlabel("X")
    ax.set_zticks(np.arange(5), labels = denormalize_min_max(np.arange(5)/5, clip_bounds[2], clip_bounds[3]).round(2), va='bottom')
    ax.set_zlabel("Y")
    
    # Seleziona le date a intervalli regolari per i tick dell'asse x
    month_indices = np.linspace(0, 103, num=8, dtype=int)
    month_labels = [f"{dates[i].strftime('%b')} {dates[i].year}" for i in month_indices]

    ax.set_yticks(month_indices)
    ax.set_yticklabels(month_labels, ha='left', va='baseline') #, rotation=45
    ax.set_ylabel("Dates", labelpad = 18)
    
        
    ax.set_title(title_ext)

    # Salvataggio o visualizzazione
    if save:
        plt.tight_layout()
        #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(save_path, dpi=400, bbox_inches="tight", transparent=True)
        plt.close(fig)
    else:
        plt.show()
    
    #plt.show()


import pickle 
import numpy as np
import copy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm, colors
import matplotlib

results = []
seed_values = [3,5,23,24,36,40,66,79,86,97]

for seed in seed_values:
    path_to_load_results = f"/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/results/01_marco_st_rise_original_stability_tot/rise_st_original_stability_seed{seed}.pkl"
    # Load della lista results
    with open(path_to_load_results, 'rb') as file:
        result = pickle.load(file)
        results.append(result)

for nr_instance in range(0, len(vottignasco_test_image)):

    data_target = vottignasco_test_dates[nr_instance].astype('datetime64[D]')

    for index_i in range(2,5):
        output_path_mean_rot1 = f"/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/scripts/notebooks/results/results_st_rise_stability_mean/#{nr_instance}_S{index_i}_mean_rot1"
        output_path_mean_rot2 = f"/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/scripts/notebooks/results/results_st_rise_stability_mean/#{nr_instance}_S{index_i}_mean_rot2"
        output_path_std_rot1  = f"/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/scripts/notebooks/results/results_st_rise_stability_std/#{nr_instance}_S{index_i}_std_rot1"
        output_path_std_rot2  = f"/leonardo_work/try25_pellegrino/Water_Resources/rise-video/XAI/spatial_temporal/scripts/notebooks/results/results_st_rise_stability_std/#{nr_instance}_S{index_i}_std_rot2"

        all_nr_instances_instances_s_i = np.zeros((10,104,5,8))
        
        for nr_result,result in enumerate(results):
            all_nr_instances_instances_s_i[nr_result] = result["saliency_videos"][nr_instance,(index_i-1),:]


        s_i_mean_all_seed_instance_nr_instance = np.mean(all_nr_instances_instances_s_i, axis=0)
        s_i_std_all_seed_instance_nr_instance  = np.std(all_nr_instances_instances_s_i, axis=0)

        plot_saliency_video_3d(s_i_mean_all_seed_instance_nr_instance, 2, high_worst=True, elev = 30,  azim = -40,
                       test_images_dates = vottignasco_test_image_dates, instance_number = nr_instance,
                       shapefile = piedmont_bounds, ROI_box = ROI_box, clip_bounds=[xmin_clip, xmax_clip, ymin_clip, ymax_clip],
                       target_shapefile = arpa_wt_sensors, title_ext= f"RISE Saliency Mean #{nr_instance}, Date Target: {data_target} - Coefficients",
                       save=True, save_path=output_path_mean_rot1)
        
        plot_saliency_video_3d(s_i_mean_all_seed_instance_nr_instance, 2, high_worst=True, elev = 35,  azim = -30,
                       test_images_dates = vottignasco_test_image_dates, instance_number = nr_instance,
                       shapefile = piedmont_bounds, ROI_box = ROI_box, clip_bounds=[xmin_clip, xmax_clip, ymin_clip, ymax_clip],
                       target_shapefile = arpa_wt_sensors, title_ext= f"RISE Saliency Mean #{nr_instance}, Date Target: {data_target} - Coefficients",
                       save=True, save_path=output_path_mean_rot2)

        plot_saliency_video_3d(s_i_std_all_seed_instance_nr_instance, 2, high_worst=True, elev = 30,  azim = -40,
                       test_images_dates = vottignasco_test_image_dates, instance_number = nr_instance,
                       shapefile = piedmont_bounds, ROI_box = ROI_box, clip_bounds=[xmin_clip, xmax_clip, ymin_clip, ymax_clip],
                       target_shapefile = arpa_wt_sensors, title_ext= f"RISE Saliency Std #{nr_instance}, Date Target: {data_target} - Coefficients",
                       save=True, save_path=output_path_std_rot1)
        
        plot_saliency_video_3d(s_i_std_all_seed_instance_nr_instance, 2, high_worst=True, elev = 30,  azim = -40,
                       test_images_dates = vottignasco_test_image_dates, instance_number = nr_instance,
                       shapefile = piedmont_bounds, ROI_box = ROI_box, clip_bounds=[xmin_clip, xmax_clip, ymin_clip, ymax_clip],
                       target_shapefile = arpa_wt_sensors, title_ext= f"RISE Saliency Std #{nr_instance}, Date Target: {data_target} - Coefficients",
                       save=True, save_path=output_path_std_rot2)

        # FUNZIONE DI PLOT 3D CHE SALVA 
        ############################# QUI ###################################################

    print(f"END {nr_instance}")