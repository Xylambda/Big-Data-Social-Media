import pandas as pd

# Read dataset
absenteeism_df = pd.read_csv("Absenteeism_at_work.csv", sep = ";")

# Transform a categorical variable to dummy variables (one hot)
dummy_var = pd.get_dummies(absenteeism_df["Day of the week"])

# Add dummy_var to the original dataset
absenteeism_df = pd.concat([absenteeism_df, dummy_var], axis = 1)

# Remove old column to prevent errors
absenteeism_df = absenteeism_df.drop("Day of the week", axis = 1)

# See the results
print(absenteeism_df.head())

# ===================== CREDITS =====================
### Aletta Smits:
# Big Data and Social Media / Data Learning Class (week 1 - Day 2)

### Rowan Langford:
# https://towardsdatascience.com/the-dummys-guide-to-creating-dummy-variables-f21faddb1d40

### Shanelynn:
# https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
