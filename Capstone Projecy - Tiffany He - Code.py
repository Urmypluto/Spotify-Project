import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

# Load the Spotify dataset
file_path = '/Users/tiffanymacair/Desktop/spotify52kData.csv'
data = pd.read_csv(file_path)

# [Preprocessing Data]
# Checking for missing values
print("\nPreprocessing data:")
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Histograms for numeric features
print("\nHistograms for numeric features: ")
numeric_features = ['popularity', 'duration', 'danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
data[numeric_features].hist(bins=15, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Features")
plt.show()

# Create a copy of the data to avoid changing the original dataframe
data_encoded = data.copy()

# Label encode categorical features
label_encoder = LabelEncoder()
categorical_features = data_encoded.select_dtypes(include=['object', 'category']).columns

# Apply label encoding to each categorical column
for column in categorical_features:
    data_encoded[column] = label_encoder.fit_transform(data_encoded[column])

# Correlation heatmap
print("\nCorrelation heatmap: ")
correlation_matrix = data_encoded.corr()
plt.figure(figsize=(12, 10))  # Adjust the size as needed
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='RdPu', square=True)
plt.title("Correlation Heatmap of All Features")
plt.show()


# Q1
print("\nQ1")

# Selecting the 10 features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Plotting histograms
plt.figure(figsize=(20, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 5, i+1)
    sns.histplot(data[feature], kde=True)
    plt.title(feature)
plt.tight_layout()
plt.show()


# Q2
print("\nQ2")

# Convert duration from milliseconds to minutes
data['duration_min'] = data['duration'] / 60000

# Generate the scatterplot and plot the best-fit line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration_min', y='popularity', data=data)
plt.plot(data['duration_min'], np.poly1d(np.polyfit(data['duration_min'], data['popularity'], 1))(data['duration_min']), color='red')
plt.title('Correlation between Song Length and Popularity')
plt.xlabel('Song Length (minutes)')
plt.ylabel('Popularity Score')
plt.show()

# Calculate and print the Pearson correlation coefficient
pearson_corr = data['duration_min'].corr(data['popularity'])
print(f'The Pearson correlation coefficient between song length and popularity is: {pearson_corr:.4f}')

# Calculate and print the Spearman correlation coefficient
spearman_corr = data['duration_min'].corr(data['popularity'], method='spearman')
print(f'The Spearman correlation coefficient between song length and popularity is: {spearman_corr:.4f}')


# Q3
print('\nQ3')

# Group data into explicit and non-explicit
explicit_data = data[data['explicit'] == True]['popularity']
non_explicit_data = data[data['explicit'] == False]['popularity']

# Calculate variance for both groups
var_explicit = explicit_data.var()
var_non_explicit = non_explicit_data.var()

# Perform the independent t-test
t_stat, p_value = ttest_ind(explicit_data, non_explicit_data)

# Adjust p-value for a one-tailed test
p_value /= 2

# Print variance and t-test results
print(f'Variance of explicit songs: {var_explicit:.4f}')
print(f'Variance of non-explicit songs: {var_non_explicit:.4f}')
print(f'Independent one-tailed t-test statistic: {t_stat:.4f}')
print(f'Independent one-tailed p-value: {p_value:.4f}')


# Q4
print('\nQ4')

# Group data into major and minor key
major_key_data = data[data['mode'] == 1]['popularity']
minor_key_data = data[data['mode'] == 0]['popularity']

# Calculate variance for both groups
var_major = major_key_data.var()
var_minor = minor_key_data.var()

# Perform the independent t-test
t_stat, p_value = ttest_ind(major_key_data, minor_key_data)

# Adjust p-value for a one-tailed test
p_value /= 2

# Print variance and t-test results
print(f'Variance of major key songs: {var_major:.4f}')
print(f'Variance of minor key songs: {var_minor:.4f}')
print(f'Independent one-tailed t-test statistic: {t_stat:.4f}')
print(f'Independent one-tailed p-value: {p_value:.4f}')


# Q5
print('\nQ5')

# Create a scatterplot for energy vs loudness
plt.figure(figsize=(10, 6))
sns.scatterplot(x='energy', y='loudness', data=data)
plt.plot(data['energy'], np.poly1d(np.polyfit(data['energy'], data['loudness'], 1))(data['energy']), color='red')
plt.title('Energy vs Loudness of Songs')
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
plt.show()

# Calculate the Pearson correlation coefficient
correlation_coef, p_value = pearsonr(data['energy'], data['loudness'])
print(f'Pearson correlation coefficient between energy and loudness: {correlation_coef:.4f}')


# Q6
print('\nQ6')

# List of features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Store R-squared values for each feature
r2_scores = {}

plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Loop through each feature and fit a linear regression model
for i, feature in enumerate(features):
    plt.subplot(2, 5, i+1)

    # Scatter plot
    plt.scatter(data[feature], data['popularity'], alpha=0.3)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(data[[feature]], data['popularity'])
    y_pred = model.predict(data[[feature]])

    # Regression line
    plt.plot(data[feature], y_pred, color='red')
    plt.title(feature)
    
    # Evaluate the model using R-squared
    r2 = r2_score(data['popularity'], y_pred)
    r2_scores[feature] = r2
    print(f'{feature}: R-squared = {r2:.4f}')

# Find the best feature with the highest R-squared value
best_feature = max(r2_scores, key=r2_scores.get)
print(f'\nBest feature predicting popularity: {best_feature} with R-squared = {r2_scores[best_feature]:.4f}')

# Show the plot
plt.show()


# Q7
print('\nQ7')

# Features and target
X = data[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = data['popularity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17754923)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'Model R-squared: {r2:.4f}')
print(f'Model RMSE: {rmse:.4f}')


# Q8
print('\nQ8')

# Standardize the features
X_std = StandardScaler().fit_transform(data[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
                                             'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
# Perform PCA on the standardized data
pca = PCA().fit(X_std)

# Plot Eigenvalues on Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--')  # Kaiser criterion (eigenvalue=1)
plt.xticks(range(1, len(pca.explained_variance_) + 1))
plt.show()

# Calculate the fourth eigenvalue
fourth_eigenvalue = pca.explained_variance_[3]
print(f"\nThe fourth eigenvalue is: {fourth_eigenvalue:.4f}")

# Proportion of variance explained by the first three principal components
variance_three_components = pca.explained_variance_ratio_[:3]
cumulative_var_three_components = np.cumsum(variance_three_components)
print(f'Cumulative variance explained by the first three components: {cumulative_var_three_components[-1]:.4f}\n')

# Perform PCA again using the first three principal components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)

# Determine the optimal number of clusters for K-means using silhouette scores
range_n_clusters = range(2, 11)  
silhouette_avg_scores = []

for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=17754923)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    # The silhouette score gives the average value for all samples.
    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    print(f"For n_clusters = {num_clusters}, the average silhouette_score is : {silhouette_avg:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
plt.title('Silhouette Method for Determining the Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.xticks(range_n_clusters)
plt.show()

optimal_clusters = range_n_clusters[np.argmax(silhouette_avg_scores)]
print(f"\nThe optimal number of clusters based on silhouette score is: {optimal_clusters}")

# Apply DBSCAN to determine the optimal number of clusters
dbscan = DBSCAN(eps=0.5, min_samples=5) # commonly used value for eps and min_samples
clusters = dbscan.fit_predict(X_pca)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f'The optimal number of clusters based on DBSCAN is: {n_clusters}')



# Q9
print('\nQ9')

# Features and target
X = data[['valence']]  # Predictor - valence
y = data['mode']  # Target - key mode (major=1, minor=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17754923)

# Initialize the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(f'Coefficient for valence: {log_reg.coef_[0][0]:.4f}')  
print(f'Intercept: {log_reg.intercept_[0]:.4f}')  

# Make predictions on the testing data
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Model Accuracy: {accuracy:.4f}')
print(f'\nClassification Report: \n{class_report}')
print(f'ROC-AUC Score: {roc_auc:.4f}')

# Plotting the logistic regression probabilities
plt.figure(figsize=(10, 6))
plt.scatter(X_train['valence'], y_train, color='black', zorder=20, label='Training Data')
valence_values = np.linspace(X['valence'].min(), X['valence'].max(), 300)
valence_values_df = pd.DataFrame(valence_values, columns=['valence'])
probabilities = log_reg.predict_proba(valence_values_df)[:, 1]

# Plot probability curve (Sigmoid function)
plt.plot(valence_values, probabilities, color='red', linewidth=3, label='Probability Curve')

# Plot the decision boundary
decision_boundary = -log_reg.intercept_[0] / log_reg.coef_[0][0]
plt.axvline(x=decision_boundary, color='blue', linestyle='--')

# Labeling the plot
plt.ylabel('Probability of Major Key')
plt.xlabel('Valence')
plt.title('Logistic Regression: Probability of Major Key by Valence')
plt.xlim(X['valence'].min(), X['valence'].max())
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()

major_key_count = data['mode'].sum() # Number of songs in major keys
minor_key_count = len(data) - major_key_count # Number of songs in minor keys
print(f'\nNumber of songs in major keys: {major_key_count}')
print(f'Number of songs in minor keys: {minor_key_count}')

# Perform another logistic regression using the "key" feature as the predictor and "mode" as the target
X = data[['key']]  # Predictor - key
y = data['mode']  # Target - mode (major=1, minor=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17754923)  

# Initialize the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(f'\nCoefficient for key: {log_reg.coef_[0][0]:.4f}')
print(f'Intercept: {log_reg.intercept_[0]:.4f}')

# Make predictions on the testing data
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Model Accuracy: {accuracy:.4f}')
print(f'\nClassification Report: \n{class_report}')
print(f'ROC-AUC Score: {roc_auc:.4f}')


# Q10
print('\nQ10')

# Convert genre labels to numerical labels
label_encoder = LabelEncoder()
data['genre_numerical'] = label_encoder.fit_transform(data['track_genre'])
y = data['genre_numerical']  # Target variable

# Option 1: Using original song features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = data[features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17754923)

# Initialize and train the random forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predict and evaluate the model using original features
y_pred = rf_model.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred)
print("Classification Report using Original Features:")
print(classification_report(y_test, y_pred))
print("Accuracy using Original Features: ", f"{accuracy_original:.4f}")


# Option 2: Using principal components
pca = PCA(n_components=3)  # Using three principal components
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=17754923)

# Initialize and train the random forest classifier for Option 2
rf_model_pca = RandomForestClassifier()
rf_model_pca.fit(X_train_pca, y_train_pca)

# Predict and evaluate the model using principal components
y_pred2 = rf_model_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred2)
print("\nClassification Report using Principal Components:")
print(classification_report(y_test_pca, y_pred2))
print("Accuracy using Principal Components: ", f"{accuracy_pca:.4f}")
