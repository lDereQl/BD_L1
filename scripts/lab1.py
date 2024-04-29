import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df_train = pd.read_csv("train_20240206.csv")

# Selected features
selected_continuous = ['LotArea', 'SalePrice', '1stFlrSF', 'GrLivArea', 'LotFrontage']
selected_categorical = ['HouseStyle', 'Street', 'Neighborhood']

# Select relevant features
df_selected = df_train[selected_continuous + selected_categorical]

# Handle missing values
for feature in selected_continuous:
    df_selected.loc[:, feature] = df_selected[feature].fillna(df_selected[feature].mean())

for feature in selected_categorical:
    df_selected.loc[:, feature] = df_selected[feature].fillna(df_selected[feature].mode()[0])


# Data Quality Report
data_quality_report = df_selected.describe(include='all').T

# Visualizations
plt.figure(figsize=(10, 8))

# Correlation heatmap for continuous features
sns.heatmap(df_selected[selected_continuous].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Continuous Features)')
plt.show()

# Boxplots and Pairplot
for feature in selected_continuous + selected_categorical:
    plt.figure(figsize=(15, 6))

    # Boxplot for categorical features
    if feature in selected_categorical:
        plt.subplot(1, 2, 1)
        sns.boxplot(x=feature, y='LotArea', data=df_selected)
        plt.title(f'Boxplot of {feature} vs LotArea')
        plt.xticks(rotation=45)

    # Pairplot for continuous features
    else:
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=feature, y='LotArea', data=df_selected, alpha=0.5)
        plt.title(f'{feature} vs LotArea')

    # Distribution plot for both types of features
    plt.subplot(1, 2, 2)
    sns.histplot(df_selected[feature], bins=20, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Display Data Quality Report
print("Data Quality Report:")
print(data_quality_report)
