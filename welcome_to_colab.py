import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = '/content/sample_data/Electric_Vehicle_Population_Size_History_By_County_.csv'
data = pd.read_csv(file_path)

# Convert "Date" to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Convert numeric columns to integers (removing commas where necessary)
columns_to_clean = [
    "Battery Electric Vehicles (BEVs)",
    "Plug-In Hybrid Electric Vehicles (PHEVs)",
    "Electric Vehicle (EV) Total",
    "Non-Electric Vehicle Total",
    "Total Vehicles",
]

for col in columns_to_clean:
    data[col] = pd.to_numeric(data[col].str.replace(",", ""), errors='coerce')

# Display dataset head and description
print("Dataset Head:")
print(data.head())

print("\nDataset Description:")
print(data.describe())

# Feature engineering: Extracting year, month, and day of the week
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Define features (X) and target (y)
X = data[['Year', 'Month', 'DayOfWeek']]
y = data['Electric Vehicle (EV) Total']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')

# Box Plot for Electric Vehicle Count by Day of the Week
plt.figure(figsize=(10, 6))
sns.boxplot(x='DayOfWeek', y='Electric Vehicle (EV) Total', data=data)
plt.title('Electric Vehicle Count by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Electric Vehicle Count')
plt.show()

# Histogram for Electric Vehicle Count distribution
plt.figure(figsize=(10, 6))
plt.hist(data['Electric Vehicle (EV) Total'], bins=30, edgecolor='k')
plt.title('Distribution of Electric Vehicle Count')
plt.xlabel('Electric Vehicle Count')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot for predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Month'], y_test, label='True Count')
plt.scatter(X_test['Month'], y_pred, label='Predicted Count', marker='x')
plt.xlabel('Month')
plt.ylabel('Electric Vehicle Count')
plt.legend()
plt.title('True vs Predicted Electric Vehicle Count')
plt.show()
