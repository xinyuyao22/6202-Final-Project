import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Training data
app_train = pd.read_csv('application_train.csv')
print('Training data shape: ', app_train.shape)
print(app_train.head())
# Testing data features
app_test = pd.read_csv('application_test.csv')
print('Testing data shape: ', app_test.shape)
print(app_test.head())

target = app_train['TARGET']
plt.hist(target.values)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
print((app_train['DAYS_BIRTH'] / 365).describe())
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages')
plt.show()
'''
The target == 1 curve skews towards the younger end of the range. Although this is not a significant correlation (-0.07 correlation coefficient),
this variable is likely going to be useful in a machine learning model because it does affect the target. Let's look at this relationship in 
another way: average failure to repay loans by age bracket.

To make this graph, first we cut the age category into bins of 5 years each. Then, for each bin, we calculate the average value of the target,
which tells us the ratio of loans that were not repaid in each age category.
'''
# Age information into a separate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
print(age_data.head())
# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
print(age_groups)
plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')
plt.show()

print(app_train['DAYS_EMPLOYED'].describe())
# The maximum value (besides being positive) is about 1000 years
plt.hist(app_train['DAYS_EMPLOYED'].values)
plt.title('Days Employment Histograms')
plt.xlabel('Days Employment')
plt.ylabel('Frequency')
plt.show()
anom = app_train[app_train['DAYS_EMPLOYED'] > 36500]
non_anom = app_train[app_train['DAYS_EMPLOYED'] < 36500]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
plt.hist(app_train['DAYS_EMPLOYED'].values)
plt.title('Days Employment Histograms')
plt.xlabel('Days Employment')
plt.ylabel('Frequency')
plt.show()
app_train['DAYS_EMPLOYED'] = abs(app_train['DAYS_EMPLOYED'])
app_test['DAYS_EMPLOYED'] = abs(app_test['DAYS_EMPLOYED'])
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
# Function to calculate missing values by column# Funct
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
# Missing values statistics
print(missing_values_table(app_train))
print(missing_values_table(app_test))
for df in [app_train,app_test]:
    for col in ['COMMONAREA_MEDI', 'COMMONAREA_AVG', 'COMMONAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MODE',
                'NONLIVINGAPARTMENTS_AVG', 'FONDKAPREMONT_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAPARTMENTS_AVG',
                'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'FLOORSMIN_AVG', 'LANDAREA_MODE', 'LANDAREA_MEDI', 'LANDAREA_AVG', 'DEF_30_CNT_SOCIAL_CIRCLE',
                'DEF_60_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_HOUR',
                'AMT_REQ_CREDIT_BUREAU_YEAR', 'TOTALAREA_MODE', 'BASEMENTAREA_MODE', 'BASEMENTAREA_MEDI', 'BASEMENTAREA_AVG', 'NONLIVINGAREA_MODE',
                'NONLIVINGAREA_MEDI', 'NONLIVINGAREA_AVG', 'ELEVATORS_AVG', 'ELEVATORS_MODE', 'ELEVATORS_MEDI', 'APARTMENTS_MODE', 'APARTMENTS_MEDI',
                'APARTMENTS_AVG', 'ENTRANCES_MEDI', 'ENTRANCES_MODE', 'ENTRANCES_AVG', 'LIVINGAREA_MEDI', 'LIVINGAREA_MODE', 'LIVINGAREA_AVG',
                'FLOORSMAX_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMAX_MEDI', 'DAYS_EMPLOYED']:
        df[col].fillna(0, inplace=True)
    df.drop(['NAME_TYPE_SUITE', 'WALLSMATERIAL_MODE', 'HOUSETYPE_MODE'],axis=1, inplace=True)
    df['OCCUPATION_TYPE'].fillna('unemployed', inplace=True)
    df['EMERGENCYSTATE_MODE'].fillna('No', inplace=True)
    df['NAME_CONTRACT_TYPE'].fillna('Cash loans', inplace=True)
# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
# Need to impute missing values
app_train = imputer.fit_transform(app_train)
app_test = imputer.fit_transform(app_test)

# Find correlations with the target and sort
correlations = pd.DataFrame(app_train).corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))