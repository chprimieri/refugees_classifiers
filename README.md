# Refugees Estimator

Apply machine learning algorithms to predict the number of refugees that will move from one country to another, using geographic distances and regions and economic and social indicators.

## Description

ML Algorithms:
- Random Forest
- Ada Boost
- Gradient Boosting

Datasets:
- UNHCR refugees data: https://www.unhcr.org/refugee-statistics/download
- UNDP development indicators: https://hdr.undp.org/data-center/documentation-and-downloads
- UNStats countries and regions: https://unstats.un.org/unsd/methodology/m49/overview/
- CEPII geographic distances: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6 

### Executing program

* Generate the data.csv dataset:
```
python3 data_cleaning.py
```

* Run the ML algorithms
```
python3 main.py
```

## Authors

Camila Haas Primieri - @chprimieri
