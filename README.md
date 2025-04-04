## Method Overview

Spatio-temporal deep learning models are powerful tools for tasks like environmental monitoring, but their black-box nature limits interpretability. To address this, we adapt three popular model-agnostic XAI methods—**LIME**, **SHAP**, and **RISE**—to generate **spatial**, **temporal**, and **spatio-temporal** explanations for geospatial image time series.

These extensions enable a detailed understanding of model behavior across space and time. We evaluate the methods on a real-world task: **groundwater level prediction**. Our results show that each technique offers unique insights, highlighting the value of combining different explanation strategies in dynamic environmental contexts.

![algorithm_pipelines](https://github.com/user-attachments/assets/35a8897d-c76f-4d70-b999-e7d8bbabbfa6)


## Repository Contents

- `spatial/`  
  Extensions of RISE, LIME, and SHAP for spatial explanations, along with corresponding results.

- `temporal/`  
  Scripts and outputs for temporal explanations.

- `spatial_temporal/`  
  Code and results for spatio-temporal explanations.
