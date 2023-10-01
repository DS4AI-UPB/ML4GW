# ML4GW

Efficient Machine Learning Ensemble Methods for Detecting Gravitational Wave Glitches in LIGO Time Series

## Article:

Elena-Simona Apostol, Ciprian-Octavian Truică. *Efficient Machine Learning Ensemble Methods for Detecting Gravitational Wave Glitches in LIGO Time Series*. ICCP, 2023. DOI: [https://doi.org/TBA](https://doi.org/TBA)

## Packages

Python >= 3.9
- SciKit-Learn
- XGBoost
- Pandas
- numpy
- matplotlib
- SciPy
- tensorflow
- Keras
- 
## Utilization

To run ShallowWave

`python ShallowWave.py FILE_NAME`

To run DeepWave

`python DeepWave.py FILE_NAME`

The FILE_NAME is a csv file with the followind columns \['id', 'duration', 'peak_freq', 'central_freq', 'bandwidth', 'snr', 'label'\].
