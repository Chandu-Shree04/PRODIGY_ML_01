import pandas as pd
from sklearn.linear_model import LinearRegression

# Load training and test datasets
train_data = pd.read_csv(r'C:/Users/Admin/Downloads/train.csv')
test_data = pd.read_csv(r'C:/Users/Admin/Downloads/test.csv')

# Extracting features and target variable
X_train = train_data[['LotArea','BedroomAbvGr','FullBath','HalfBath']]
print(X_train)
y_train = train_data['SalePrice']
print(y_train)
X_test = test_data[['LotArea','BedroomAbvGr','FullBath','HalfBath']]


# Initialize linear regression model
model = LinearRegression()

# Fit the model on training data
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
print(y_pred)
test_data['Predicted sale Price']=y_pred
test_data.to_csv('new.csv', index=False)
print("Prediction Saved")
