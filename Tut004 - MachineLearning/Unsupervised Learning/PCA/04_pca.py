"""
Unsupervised Learning - Algorithmus 2: Hauptkomponentenanalyse (PCA)
Datensatz: Wein 

Ziel: Die 13 Messwerte sollen auf wenige neue Größen verdichtet werden,
ohne dass dabei viel Information verloren geht. Auch hier gibt es keine
Zielgröße, PCA sucht allein die Richtungen, in denen die Daten am
stärksten streuen.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")			 # Diagramme als Datei speichern statt anzeigen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd			 # Tabelle mit benannten Spalten erstellen
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ORDNER = Path(__file__).parent		 # Ordner dieses Skripts, Diagramme landen daneben

# 1. Daten laden und skalieren
# PCA sucht Richtungen größter Streuung. Ohne Skalierung würde der
# Messwert mit der größten Zahlenskala, hier Prolin, die erste
# Hauptkomponente allein bestimmen.
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

X = pd.DataFrame(daten.data, columns=daten.feature_names).rename(columns=SPALTEN)
echte_sorte = daten.target		 # nur zur Einfärbung des Diagramms, PCA kennt die Sorten nicht
X_skaliert = StandardScaler().fit_transform(X)	 # alle Messwerte auf Mittelwert 0 und Streuung 1 bringen

print("Ausgangslage:", X.shape[1], "Messwerte je Weinprobe")
print()

# 2. Alle Hauptkomponenten berechnen
# Jede Hauptkomponente ist eine gewichtete Summe der ursprünglichen
# Messwerte. Sie sind nach erklärter Streuung geordnet und stehen
# senkrecht aufeinander, enthalten also keine doppelte Information.
pca_voll = PCA().fit(X_skaliert)	 # ohne n_components werden alle 13 Komponenten berechnet
anteil = pca_voll.explained_variance_ratio_	 # Anteil je Komponente an der Gesamtstreuung, Summe ergibt 1,0
kumuliert = np.cumsum(anteil)		 # laufende Summe, zeigt was die ersten i Komponenten zusammen abdecken

print("Erklaerte Streuung je Hauptkomponente:")
for i in range(5):
    print(f"  HK{i+1}: {anteil[i]*100:5.1f} %   kumuliert: {kumuliert[i]*100:5.1f} %")
print(f"  ... ab HK6 tragen alle weiteren zusammen nur noch "
      f"{(1-kumuliert[4])*100:.1f} % bei")
print()

noetig = int(np.argmax(kumuliert >= 0.80) + 1)	 # argmax findet die erste Komponente, ab der 80 Prozent erreicht sind, plus 1 weil Positionen bei 0 zählen
print(f"Fuer 80 % der Gesamtstreuung genuegen {noetig} von {X.shape[1]} Komponenten.")
print()

# 3. Auf zwei Komponenten reduzieren
pca = PCA(n_components=2)		 # behält nur die beiden wichtigsten Richtungen
X_reduziert = pca.fit_transform(X_skaliert)	 # projiziert die Daten darauf, aus 178 x 13 wird 178 x 2
print("Nach der Reduktion:", X_reduziert.shape[1], "Spalten statt", X.shape[1],
      f"- erhalten bleiben {kumuliert[1]*100:.1f} % der Streuung")
print()

# 4. Woraus bestehen die Hauptkomponenten
# Die Ladungen zeigen, welche Messwerte eine Komponente prägen.
ladungen = pd.DataFrame(pca.components_.T, index=X.columns,
                        columns=["HK1", "HK2"])  # components_ enthält die Gewichte, .T dreht die Tabelle damit die Messwerte in den Zeilen stehen
print("Staerkste Beitraege zur ersten Hauptkomponente:")
print(ladungen["HK1"].abs().sort_values(ascending=False).head(4).round(3).to_string())  # abs weil nur die Stärke zählt, nicht das Vorzeichen
print()

# 5. Nutzen für nachgelagerte Verfahren
# k-Means wird einmal auf allen 13 Messwerten und einmal auf den zwei
# Hauptkomponenten gerechnet. Der Vergleich zeigt, ob die Reduktion die
# Gruppenstruktur erhält.
gruppe_voll = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_skaliert)
gruppe_pca = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_reduziert)

print("k-Means auf 13 Messwerten, Uebereinstimmung mit den Sorten:",
      round(adjusted_rand_score(echte_sorte, gruppe_voll), 3))
print("k-Means auf  2 Hauptkomponenten, Uebereinstimmung:          ",
      round(adjusted_rand_score(echte_sorte, gruppe_pca), 3))

# 6. Diagramme
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(anteil) + 1), anteil * 100, color="#2c4a7c", label="einzeln")  # Balken zeigen den Beitrag jeder einzelnen Komponente
plt.plot(range(1, len(anteil) + 1), kumuliert * 100, "o-", color="#b5762a",
         label="kumuliert")		 # Linie zeigt die laufende Summe
plt.axhline(80, color="grey", linestyle="--", linewidth=1)  # Hilfslinie bei 80 Prozent
plt.xlabel("Hauptkomponente")
plt.ylabel("Erklaerte Streuung in Prozent")
plt.title("PCA: Informationsgehalt der Komponenten")
plt.legend()
plt.tight_layout()
plt.savefig(ORDNER / "04_streuung.png", dpi=120)

plt.figure(figsize=(6.5, 5))
for s in range(3):			 # zeichnet jede Sorte einzeln für eigene Farbe und Legendeneintrag
    plt.scatter(X_reduziert[echte_sorte == s, 0], X_reduziert[echte_sorte == s, 1],  # Spalte 0 ist HK1, Spalte 1 ist HK2
                label=SORTEN[s], alpha=0.75, edgecolor="k")
plt.xlabel(f"Hauptkomponente 1 ({anteil[0]*100:.1f} %)")
plt.ylabel(f"Hauptkomponente 2 ({anteil[1]*100:.1f} %)")
plt.title("13 Messwerte auf zwei Komponenten verdichtet")
plt.legend()
plt.tight_layout()
plt.savefig(ORDNER / "04_projektion.png", dpi=120)
print("\nDiagramme gespeichert: 04_streuung.png, 04_projektion.png")
