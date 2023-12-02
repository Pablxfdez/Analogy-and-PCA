# PCA and Analogy in Computational Geometry Project

## Introduction
This project focuses on analyzing changes in temperature and geopotential height of the atmosphere using NetCDF files from the years 2021 and 2022. The study centers on identifying principal components and studying the most analogous days based on Euclidean distance.

## Project Description
The practice involves:

- **Analyzing Atmospheric Data**: Utilizing NetCDF files for temperature and geopotential height data provided by NCEP climatological re-analysis for the years 2021 and 2022.
- **Principal Component Analysis (PCA)**: Implementing PCA to identify the main components that influence the atmospheric system's state.
- **Analogy Study**: Determining the most analogous days by calculating the Euclidean distance, focusing on specific atmospheric variables.

## Materials and Methods
The project was approached using computational techniques and Python libraries:

- **NetCDF Files**: Four files were used, two each for temperature (air*) and geopotential height (hgt*), corresponding to 2021 and 2022.
- **Python Libraries**: Numpy for calculations, Matplotlib.plot for visualization, and Random library for generating initial x0 values from a uniform distribution over the interval (0,1).
- **Data Organization**: The data is organized into 144 longitudes (x), 73 latitudes (y), and 17 pressure levels (p), with time (t) discretized into daily lapses.

The code is based on the template provided: GCOM2023-practica_PCA_ANN_plantilla.py.

## Repository Contents

1. **FernandezdelAmoP_PCAyAnalogia.pdf**
   - **Description**: This document provides a comprehensive overview of the project, including theoretical background, methodology, and explanation of the computational techniques used in the analysis.

2. **FernandezdelAmoP_PCAyAnalogia.py**
   - **Description**: The Python script is the core of the project, containing the implementation of the PCA and analogy analysis. It includes the calculation of principal components and the determination of the most analogous days.

## Conclusion
Through this practice, a detailed analysis of changes in temperature and geopotential height of the atmosphere was achieved. The identification of the four principal components highlighted the significance of latitude in calculating geopotential height. The most analogous days to a given day (a0) were determined based on Euclidean distance, providing a solid framework for future climatological analyses. The results demonstrate the efficiency of this study, as the difference between expected and actual values is minimal. This study also underscored the importance and utility of climatological data in understanding changes in our environment.
