# Diploma_thesis_scripts
This repository contains a complete pipeline for processing Sentinel-3 satellite data and analyzing the Urban Heat Island (UHI) effect through statistical analysis and machine learning models.

Project Structure & File Descriptions:
1. Sentinel3_preprocessing.py 
- Handles cleaning and preprocessing of Sentinel-3 satellite data.
- Filters, normalizes, and transforms raw data to make it suitable for analysis.

2. Sentinel3_clip.py 
- Applies clipping to the Sentinel-3 data, restricting it to specific geographical areas of interest.
- Focuses on urban centers and predefined regions for UHI analysis.
  
3. Split_Data_daytime_and_nighttime.py 
- Splits the dataset into daytime and nighttime measurements.
- Enables comparison of UHI intensity during different times of the day.
  
4. UHI_Statistics.py 
- Computes statistical metrics related to the Urban Heat Island (UHI) effect.
- Analyzes temperature differences between urban and rural areas, identifying trends and patterns.

5. model_classification.py 
- Implements classification models to categorize regions based on their thermal behavior.
- Used to identify areas with high or low UHI impact.

6. model_regression.py (The model did not perform perfectly)
- Contains regression models to predict temperature values based on environmental factors.
- Helps estimate temperature variations in specific locations.
  
7. model_inference.py 
- Provides inference functionality for applying trained models to new data.
- Used to generate predictions on real-world or unseen datasets.
