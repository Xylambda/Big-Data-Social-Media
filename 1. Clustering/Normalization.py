import numpy as np
import pandas as pd

from sklearn import preprocessing


# ===================== TYPERS OF TRANFORMATIONS =====================
my_data = [1267, 7, 5432, 987, 1703, 123, 9098, 4072, 540, 3078, 54]

# ---- Logarithmic Transformation ----
log_data = np.log(my_data)


# ---- Normalization: values between 0 and 1 ----
# This function expects a 2D array: [my_data]
normalized_data = preprocessing.normalize([my_data])

# Remove 1 dimension
normalized_data_1D = [i for i in normalized_data[0]]
    
# ---- Standardization: mean: 0, s.dev: 1 ----
standardized_data = preprocessing.scale(my_data)

print("Original data: ", my_data)
print("-------------")
print("Logarithmic Transformation: ", log_data)
print("-------------")
print("Normalized data: ", normalized_data_1D)
print("-------------")
print("Standardized data: ", standardized_data)

# ===================== EXAMPLE WITH DATASET =====================
absenteeism_df = pd.read_csv("Absenteeism_at_work.csv", sep = ";")

absenteeism_df["Age"] = preprocessing.scale(absenteeism_df["Age"].values)

print(absenteeism_df["Age"])

# ===================== CREDITS =====================
### Scikit:
# https://scikit-learn.org/stable/modules/preprocessing.html
    
### Aletta Smits:
# Big Data and Social Media / Data Learning Class (week 1 - Day 2)

### Robert R.F. DeFilippi
# https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
