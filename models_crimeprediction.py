# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifierr

# Evaluation Metrics
# from yellowbrick.classifier import ClassificationReport
from sklearn import metrics

df = pd.read_csv("finalchicago-15,16,17,23.csv")
# print(df.head())
# print(df.info())

# Preprocessing
# Remove NaN Value (As Dataset is huge, the NaN row could be neglectable)  
df = df.dropna()

# As the dataset is too huge is size, we would just subsampled a dataset for modelling as proof of concept
# df = df.sample(n=100000)

# Remove irrelevant/not meaningfull attributes
# df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['ID'], axis=1)
df = df.drop(['Case Number'], axis=1) 

# print(df.info())

# Convert 'Date' column to datetime format, letting Pandas infer the format
df['date2'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')

# Extracting Year, Month, Day, Hour, Minute, Second
df['Year'] = df['date2'].dt.year
df['Month'] = df['date2'].dt.month
df['Day'] = df['date2'].dt.day
df['Hour'] = df['date2'].dt.hour
df['Minute'] = df['date2'].dt.minute
df['Second'] = df['date2'].dt.second

# Dropping unnecessary columns
df = df.drop(['Date', 'date2', 'Updated On'], axis=1)

# print(df.head())

# Convert Categorical Attributes to Numerical
df['Block'] = pd.factorize(df["Block"])[0]
df['IUCR'] = pd.factorize(df["IUCR"])[0]
df['Description'] = pd.factorize(df["Description"])[0]
df['Location Description'] = pd.factorize(df["Location Description"])[0]
df['FBI Code'] = pd.factorize(df["FBI Code"])[0]
df['Location'] = pd.factorize(df["Location"])[0]

Target = 'Primary Type'
print('Target: ', Target)

# Plot Bar Chart visualize Primary Types
plt.figure(figsize=(14,10))
plt.title('Amount of Crimes by Primary Type')
plt.ylabel('Crime Type')
plt.xlabel('Amount of Crimes')

# df.groupby([df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')

# plt.show()

# At previous plot, we could see that the classes is quite imbalance
# Therefore, we are going to group several less occured Crime Type into 'Others' to reduce the Target Class amount

# First, we sum up the amount of Crime Type happened and select the last 13 classes
all_classes = df.groupby(['Primary Type'])['Block'].size().reset_index()
all_classes['Amt'] = all_classes['Block']
all_classes = all_classes.drop(['Block'], axis=1)
all_classes = all_classes.sort_values(['Amt'], ascending=[False])

unwanted_classes = all_classes.tail(13)
# print(unwanted_classes)

# After that, we replaced it with label 'OTHERS'
df.loc[df['Primary Type'].isin(unwanted_classes['Primary Type']), 'Primary Type'] = 'OTHERS'

# Plot Bar Chart visualize Primary Types
plt.figure(figsize=(14,10))
plt.title('Amount of Crimes by Primary Type')
plt.ylabel('Crime Type')
plt.xlabel('Amount of Crimes')

# df.groupby([df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')

# plt.show()

# Now we are left with 14 Class as our predictive class
Classes = df['Primary Type'].unique()
# print(Classes)

#Encode target labels into categorical variables:
df['Primary Type'] = pd.factorize(df["Primary Type"])[0] 
df['Primary Type'].unique()

# Feature Selection using Filter Method 
# Split Dataframe to target class and features
X_fs = df.drop(['Primary Type'], axis=1)
Y_fs = df['Primary Type']

#Using Pearson Correlation
plt.figure(figsize=(20,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

#Correlation with output variable
cor_target = abs(cor['Primary Type'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
# print(relevant_features)

# At Current Point, the attributes is select manually based on Feature Selection Part. 
Features = ["IUCR", "Description", "FBI Code"]
# print('Full Features: ', Features)

#Split dataset to Training Set & Test Set
x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]    #Features to train
x2 = x[Target]      #Target Class to train
y1 = y[Features]    #Features to test
y2 = y[Target]      #Target Class to test

print('Target Class        : ', Target)
print('Training Set Size   : ', x.shape)
print('Test Set Size       : ', y.shape)


# Neural Network
# Create Model with configuration 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

pipeline = make_pipeline(SimpleImputer(strategy='mean'), 
                         MLPClassifier(solver='adam', 
                         alpha=1e-5,
                         hidden_layer_sizes=(40,), 
                         random_state=1,
                         max_iter=1000))

# Model Training
pipeline.fit(X=x1,
             y=x2)

# Prediction
result = pipeline.predict(y[Features]) 

# Model Evaluation
mlp_ac_sc = accuracy_score(y2, result)

# Random Forest
# Create Model with configuration
rf_model = RandomForestClassifier(n_estimators=100, # Number of trees
                                  min_samples_split = 30,
                                  bootstrap = True, 
                                  max_depth = 50, 
                                  min_samples_leaf = 25)

# Model Training
rf_model.fit(X=x1,
             y=x2)

# Prediction
result = rf_model.predict(y[Features])

# Model Evaluation
rf_ac_sc = accuracy_score(y2, result)

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    criterion='gini',  # Criterion for splitting
    max_depth=50,       # Maximum depth of the tree
    min_samples_split=30,  # Minimum number of samples required to split an internal node
    min_samples_leaf=25    # Minimum number of samples required to be at a leaf node
)

# Model Training
dt_model.fit(X=x1,
             y=x2)

# Prediction
result = dt_model.predict(y[Features])

# Model Evaluation
dt_ac_sc = accuracy_score(y2, result)

# K-Nearest Neighbors
# Create Model with configuration
from sklearn.neighbors import KNeighborsClassifier

# Preprocessing to handle missing values
from sklearn.impute import SimpleImputer

# Replace NaN values with the mean of each feature
imputer = SimpleImputer(strategy='mean')
x1_imputed = imputer.fit_transform(x1)
y1_imputed = imputer.transform(y1)

# K-Nearest Neighbors
# Create Model with configuration 
knn_model = KNeighborsClassifier(n_neighbors=5)

# Model Training
knn_model.fit(X=x1_imputed, y=x2)

# Prediction
result = knn_model.predict(y1_imputed) 

knn_ac_sc = accuracy_score(y2, result)

# Initialize Logistic Regression model
from sklearn.linear_model import LogisticRegression

# logistic_regression_model = LogisticRegression(max_iter=1000)  # You can adjust max_iter based on convergence
pipeline = make_pipeline(SimpleImputer(strategy='mean'), LogisticRegression(max_iter=1000))
# Train the model
pipeline.fit(X=x1,y=x2)

# Predictions on the test set
result = pipeline.predict(y[Features])

# Model Evaluation
lr_ac_sc = accuracy_score(y2, result)


from tabulate import tabulate

l = [["MLP", mlp_ac_sc], ["Random Forest", rf_ac_sc], ["Decision Tree", dt_ac_sc],["KNN", knn_ac_sc],["Logistic Regression", lr_ac_sc]]
table = tabulate(l, headers=['Model', 'Accuracy'], tablefmt='orgtbl')

print(table)



