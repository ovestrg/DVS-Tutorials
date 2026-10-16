"""
Supervised Learning - Algorithmus 2: Decision Tree und Random Forest
Datensatz: Wein (in scikit-learn enthalten)

Ziel: Aus 13 chemischen Messwerten wird bestimmt, von welcher der drei
Rebsorten eine Weinprobe stammt.

Es werden zwei Verfahren verglichen: ein decision Tree
und ein Random Forest, der viele Bäume kombiniert.
"""

import matplotlib
matplotlib.use("Agg")			 # Diagramme als Datei speichern statt anzeigen
import matplotlib.pyplot as plt
import pandas as pd			 # Tabelle mit benannten Spalten erstellen
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Daten laden
# load_wine liefert die Messwerte in X und die Rebsorte in y.
daten = load_wine()

SPALTEN = {
    "alcohol": "Alkohol", "malic_acid": "Apfelsaeure", "ash": "Asche",
    "alcalinity_of_ash": "Asche-Alkalitaet", "magnesium": "Magnesium",
    "total_phenols": "Phenole", "flavanoids": "Flavonoide",
    "nonflavanoid_phenols": "Andere Phenole", "proanthocyanins": "Gerbstoffe",
    "color_intensity": "Farbtiefe", "hue": "Farbton",
    "od280/od315_of_diluted_wines": "Trubwert", "proline": "Prolin",
}
SORTEN = ["Sorte A", "Sorte B", "Sorte C"]

X = pd.DataFrame(daten.data, columns=daten.feature_names).rename(columns=SPALTEN)  # Eingangsgrößen X
y = daten.target			 # Klassennummern 0, 1, 2 für die drei Sorten

print("Datensatz:", X.shape[0], "Weinproben,", X.shape[1], "Messwerte")
print("Klassen:", ", ".join(SORTEN))
print()

# 2. Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)  # stratify hält das Verhältnis der Sorten in beiden Mengen gleich

# 3. Einzelner Entscheidungsbaum
# Der Baum sucht bei jeder Verzweigung den Messwert und die Schwelle,
# die die Proben am besten in reine Gruppen trennt. Reinheit wird über
# den Gini-Index gemessen.
baum = DecisionTreeClassifier(max_depth=3, random_state=42)  # max_depth begrenzt die Ebenen, sonst lernt der Baum auch Zufälligkeiten mit
baum.fit(X_train, y_train)		 # führt das eigentliche Lernen aus
treffer_baum = accuracy_score(y_test, baum.predict(X_test))  # zählt die richtig zugeordneten Testproben
print("Entscheidungsbaum, Trefferquote:", round(treffer_baum, 3))

# 4. Random Forest
# Jeder Baum bekommt eine Zufallsstichprobe der Daten und an jeder
# Verzweigung nur eine Zufallsauswahl der Messwerte. Die Vorhersage ist
# die Mehrheitsentscheidung aller Bäume.
wald = RandomForestClassifier(n_estimators=200, random_state=42)  # n_estimators legt die Anzahl der Bäume fest
wald.fit(X_train, y_train)
treffer_wald = accuracy_score(y_test, wald.predict(X_test))
print("Random Forest, Trefferquote: ", round(treffer_wald, 3))
print()

# 5. Stabilität prüfen
# Eine einzelne Aufteilung kann zufällig günstig ausfallen. Die
# Kreuzvalidierung wiederholt den Test mit wechselnden Aufteilungen.
cv_baum = cross_val_score(baum, X, y, cv=5)  # cv=5 teilt die Daten in fünf Teile, trainiert fünfmal und gibt fünf Trefferquoten zurück
cv_wald = cross_val_score(wald, X, y, cv=5)  # übergeben werden X und y komplett, die Funktion teilt selbst
print("Kreuzvalidierung Baum: ", cv_baum.round(3),
      "Mittel:", round(cv_baum.mean(), 3))	 # Mittelwert zeigt die typische Güte, die Spanne die Abhängigkeit von der Aufteilung
print("Kreuzvalidierung Wald: ", cv_wald.round(3),
      "Mittel:", round(cv_wald.mean(), 3))
print()

# 6. Wahrheitsmatrix
# Zeigt, welche Sorten verwechselt wurden. Die Diagonale sind die
# richtigen Zuordnungen, alles daneben sind Fehler.
matrix = confusion_matrix(y_test, wald.predict(X_test))  # Zeile = tatsächliche Sorte, Spalte = vorhergesagte Sorte
print("Wahrheitsmatrix Random Forest (Zeilen = wahr, Spalten = vorhergesagt):")
print(pd.DataFrame(matrix, index=SORTEN, columns=SORTEN).to_string())
print()

# 7. Welche Messwerte sind entscheidend
wichtigkeit = pd.Series(wald.feature_importances_,
                        index=X.columns).sort_values(ascending=False)  # feature_importances_ misst, wie stark jeder Messwert zur Trennung beiträgt, Summe ergibt 1,0
print("Wichtigste Messwerte laut Random Forest:")
print(wichtigkeit.head(5).round(3).to_string())	 # head(5) nimmt die fünf größten

# 8. Diagramme
plt.figure(figsize=(14, 7))
plot_tree(baum, feature_names=X.columns, class_names=SORTEN,
          filled=True, rounded=True, fontsize=9)  # zeichnet den Baum mit Regel, Gini-Wert und Probenzahl je Kasten
plt.title("Entscheidungsbaum (Tiefe 3)")
plt.tight_layout()			 # verhindert abgeschnittene Beschriftungen
plt.savefig("02_entscheidungsbaum.png", dpi=110)  # Auflösung

plt.figure(figsize=(7, 5))
wichtigkeit.head(8).sort_values().plot.barh(color="#2c4a7c")  # aufsteigend sortiert, damit der größte Balken oben steht
plt.xlabel("Bedeutung fuer die Klassifikation")
plt.title("Random Forest: wichtigste Messwerte")
plt.tight_layout()
plt.savefig("02_wichtigkeit.png", dpi=120)
print("\nDiagramme gespeichert: 02_entscheidungsbaum.png, 02_wichtigkeit.png")
