import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Charger les données
df = pd.read_csv("training_data.csv")

# Séparer les variables
X = df[["YearsExperience"]]
y = df["Salary"]

# Entraîner le modèle
model = LinearRegression()
model.fit(X, y)

# Sauvegarder le modèle
joblib.dump(model, "linear_model.txt")
print("Modèle entraîné et sauvegardé.")
