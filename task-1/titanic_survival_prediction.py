
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

data = pd.read_csv("C:/Users/kusum/OneDrive/Desktop/titanic/Titanic-Dataset.csv")
print("✅ First 5 rows of the dataset:")
print(data.head())
print("\n📊 Dataset Info:")
print(data.info()) 

print("\n📈 Summary Statistics:")
print(data.describe())  

print("\n🧮 Survived Value Counts:")
print(data['Survived'].value_counts()) 

sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survival Count by Gender')
plt.show()
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data['Age'].fillna(data['Age'].median(), inplace=True)

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

print("\n🧹 Preprocessed Data Sample:")
print(data.head())
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.2f}")

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'titanic_model.pkl')
print("\n💾 Model saved as 'titanic_model.pkl'")
