"""
Supervised Learning - Algorithmus 1: Lineare Regression
Datensatz: Diabetes(in scikit-learn enthalten)

Ziel: Aus zehn Patientenmerkmalen wird der Krankheitsfortschritt nach
einem Jahr vorhergesagt. Da Zielgröße eine Zahl ist , ist es ein Regressionsproblem und keine Klassifikation.
"""

import matplotlib
matplotlib.use("Agg")         		 # Diagramme als Datei speichern statt anzeigen
import matplotlib.pyplot as plt
import pandas as pd			 # tabelle mit benannten Spalten erstellen
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Daten laden 
# load_diabetes liefert die Merkmale in x und die Zielgroesse in y.
daten = load_diabetes()


SPALTEN = {
    "age": "Alter",
    "sex": "Geschlecht",
    "bmi": "BMI",
    "bp":  "Blutdruck",
    "s1":  "Cholesterin",
    "s2":  "LDL",
    "s3":  "HDL",
    "s4":  "Chol-Quotient",
    "s5":  "Blutfette",
    "s6":  "Blutzucker",
}

X = pd.DataFrame(daten.data, columns=daten.feature_names).rename(columns=SPALTEN)  #Eingangsgrößen x
y = daten.target   # Krankheitsfortschritt nach einem Jahr #Vorhersage y

print("Datensatz:", X.shape[0], "Beobachtungen,", X.shape[1], "Merkmale") #Shape erzeugt Zahlenpaar
print("Merkmale:", ", ".join(X.columns)) #join verbindet Zahlen mit Komma
print("Zielgroesse: Krankheitsfortschritt nach einem Jahr")
print()

# 2. Aufteilen in Trainings- und Testdaten 
# - random_state macht die Aufteilung reproduzierbar
X_train, X_test, y_train, y_test = train_test_split(  #train_test_split zieht Zufällige zeilen, gibt die vier Werte in der Reihenfolge zurück
    X, y, test_size=0.2, random_state=42) #random state fixiert zufallsgenerator, Zahl kann beliebig gewählt werden

print("Trainingsdaten:", len(X_train), "| Testdaten:", len(X_test))
print()

# 3. Modell trainieren 
# Die lineare Regression sucht die Gewichte, bei denen die Summe der
# quadrierten Abweichungen zwischen Vorhersage und Wahrheit minimal ist.
modell = LinearRegression()
modell.fit(X_train, y_train) # führt das eigentliche Lernen aus, Methode der kleinsten quadrate wird angewendet

# 4. Vorhersage und Bewertung 
y_vorhersage = modell.predict(X_test) #Textmerkmale werden in gelernte Gleichung eingesetzt, für jede der 89 Testzeilen wird ein Wert geliefert 

# MAE: durchschnittlicher Fehler der Zielgröße
# R2:  Anteil der erklärten Streuung, 1,0 wäre bspw. eine perfekte Vorhersage
print("Mittlerer absoluter Fehler:", round(mean_absolute_error(y_test, y_vorhersage), 2))
print("Bestimmtheitsmass R2:      ", round(r2_score(y_test, y_vorhersage), 3)) #setzt Fehler des Modells ins Verhältnis zu dem Fehler den man hätte wenn man mit Mittelwert vorhersagen würde. 1,0 würde heißten dass das Modell immer perfekt trifft
print()

# 5. Einfluss der einzelnen Merkmale 
# Die Koeffizienten zeigen, wie stark jedes Merkmal in die Vorhersage
# eingeht
koeffizienten = pd.Series(modell.coef_, index=X.columns).sort_values()
print("Einfluss der Merkmale (Koeffizienten):")
print(koeffizienten.round(1).to_string())

# 6. Ergebnis als Diagramm 
# Jeder Punkt ist ein Testfall. Je näher die Punkte an der Diagonalen
# liegen, desto besser trifft die Vorhersage zu

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_vorhersage, alpha=0.6, edgecolor="k") #zeichnet Punkte
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=1.5) # Erzeugung einer Hilfslinie
plt.xlabel("Tatsaechlicher Wert")
plt.ylabel("Vorhergesagter Wert")
plt.title("Lineare Regression: Vorhersage gegen Wahrheit")
plt.tight_layout() #einhaltung Format
plt.savefig("01_regression.png", dpi=120) #Auflösung
print("\nDiagramm gespeichert: 01_regression.png")
