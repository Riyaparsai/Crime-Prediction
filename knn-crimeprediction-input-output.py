
#Preprocessing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("finalchicago-15,16,17,23.csv")

# Preprocessing
# Remove NaN Value (As Dataset is huge, the NaN row could be neglectable)  
df = df.dropna()

# Remove irrelevant/not meaningfull attributes
# df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['ID'], axis=1)
df = df.drop(['Case Number'], axis=1) 

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

# Convert Categorical Attributes to Numerical
df['Block'] = pd.factorize(df["Block"])[0]
df['IUCR'] = pd.factorize(df["IUCR"])[0]
df['Description'] = pd.factorize(df["Description"])[0]
df['Location Description'] = pd.factorize(df["Location Description"])[0]
df['FBI Code'] = pd.factorize(df["FBI Code"])[0]
df['Location'] = pd.factorize(df["Location"])[0]

Target = 'Primary Type'

# First, we sum up the amount of Crime Type happened and select the last 13 classes
all_classes = df.groupby(['Primary Type'])['Block'].size().reset_index()
all_classes['Amt'] = all_classes['Block']
all_classes = all_classes.drop(['Block'], axis=1)
all_classes = all_classes.sort_values(['Amt'], ascending=[False])

unwanted_classes = all_classes.tail(13)

# After that, we replaced it with label 'OTHERS'
df.loc[df['Primary Type'].isin(unwanted_classes['Primary Type']), 'Primary Type'] = 'OTHERS'


# Now we are left with 14 Class as our predictive class
Classes = df['Primary Type'].unique()

#Encode target labels into categorical variables:
df['Primary Type'] = pd.factorize(df["Primary Type"])[0] 
df['Primary Type'].unique()

# Feature Selection using Filter Method 
# Split Dataframe to target class and features
X_fs = df.drop(['Primary Type'], axis=1)
Y_fs = df['Primary Type']


# At Current Point, the attributes is select manually based on Feature Selection Part. 
Features = ["Day","Month","Year","Latitude","Longitude"]

#Split dataset to Training Set & Test Set
x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]    #Features to train
x2 = x[Target]      #Target Class to train
y1 = y[Features]    #Features to test
y2 = y[Target]      #Target Class to test


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

# Dictionary mapping primary type numbers to names
primary_type_names = {
    0: 'ASSAULT', 1: 'CRIMINAL DAMAGE', 2: 'DECEPTIVE PRACTICE', 3: 'THEFT',
    4: 'MOTOR VEHICLE THEFT', 5: 'BURGLARY', 6: 'OTHER OFFENSE', 7: 'CRIMINAL TRESPASS',
    8: 'BATTERY', 9: 'NARCOTICS', 10: 'ROBBERY', 11: 'OFFENSE INVOLVING CHILDREN',
    12: 'WEAPONS VIOLATION', 13: 'PUBLIC PEACE VIOLATION', 14: 'ARSON', 15: 'OTHERS',
    16: 'SEX OFFENSE', 17: 'PROSTITUTION', 18: 'INTERFERENCE WITH PUBLIC OFFICER',
    19: 'CRIM SEXUAL ASSAULT', 20: 'CRIMINAL SEXUAL ASSAULT', 21: 'GAMBLING'
}

def predict_primary_type(day, month, year, latitude, longitude):
    # Create a feature vector from the input
    input_features = [[day, month, year, latitude, longitude]]
    
    # Use the trained model to predict the primary type
    predicted_primary_type = knn_model.predict(input_features)
    
    predicted_primary_type_name = primary_type_names[predicted_primary_type[0]]
    
    return predicted_primary_type_name


def get_user_input():
    day = int(input("Enter Day: "))
    month = int(input("Enter Month: "))
    year = int(input("Enter Year: "))
    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))
    return day, month, year, latitude, longitude


# Main function
def main():
    day_input, month_input, year_input, latitude_input, longitude_input = get_user_input()
    predicted_type = predict_primary_type(day_input, month_input, year_input, latitude_input, longitude_input)
    print("Predicted Primary Type:", predicted_type)

    
if __name__ == "__main__":
    main()

