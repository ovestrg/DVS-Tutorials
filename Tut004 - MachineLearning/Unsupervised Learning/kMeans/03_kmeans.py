"""
Unsupervised Learning - Algorithmus 1: k-Means
Datensatz: Wein (derselbe wie in Skript 02)

Ziel: Die Weinproben sollen in Gruppen eingeteilt werden, ohne dass das
Verfahren die Rebsorte kennt. 

Am Ende wird geprüft, wie gut die gefundenen Gruppen mit den echten
Rebsorten übereinstimmen.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")			 # Diagramme als Datei speichern statt anzeigen
import matplotlib.pyplot as plt
import pandas as pd			 # Tabelle mit benannten Spalten erstellen
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

ORDNER = Path(__file__).parent		 # Ordner dieses Skripts, Diagramme landen daneben

# 1. Daten laden
daten = load_wine()
SPALTEN = {
    "alcohol": "Alkohol", "malic_acid": "Apfelsaeure", "ash": "Asche",
    "alcalinity_of_ash": "Asche-Alkalitaet", "magnesium": "Magnesium",
    "total_phenols": "Phenole", "flavanoids": "Flavonoide",
    "nonflavanoid_phenols": "Andere Phenole", "proanthocyanins": "Gerbstoffe",
    "color_intensity": "Farbtiefe", "hue": "Farbton",
    "od280/od315_of_diluted_wines": "Trubwert", "proline": "Prolin",
}
X = pd.DataFrame(daten.data, columns=daten.feature_names).rename(columns=SPALTEN)
echte_sorte = daten.target		 # wird dem Algorithmus nicht übergeben, nur zur Bewertung am Ende

print("Datensatz:", X.shape[0], "Weinproben,", X.shape[1], "Messwerte")
print()

# 2. Skalieren
# k-Means misst Abstände. Prolin liegt im Bereich mehrerer hundert, der
# Farbton unter zwei. Ohne Skalierung würde Prolin die Abstände allein
# bestimmen.
X_skaliert = StandardScaler().fit_transform(X)	 # zieht je Spalte den Mittelwert ab und teilt durch die Streuung, danach Mittelwert 0 und Streuung 1
print("Beispiel vor dem Skalieren - Prolin:", round(X["Prolin"].mean(), 1),
      "| Farbton:", round(X["Farbton"].mean(), 2))
print("Nach dem Skalieren liegen alle Messwerte um 0 mit Streuung 1.")
print()

# 3. Passende Anzahl Gruppen suchen
# k-Means braucht die Anzahl der Gruppen als Vorgabe. Zwei Kennzahlen
# helfen bei der Wahl:
#   Inertia    - Summe der quadrierten Abstände zum Gruppenmittel, sinkt
#                immer weiter, der Knick im Verlauf ist der Hinweis
#                (Ellenbogenmethode)
#   Silhouette - wie klar die Gruppen getrennt sind, nahe 1 ist gut
print("Suche nach der passenden Gruppenzahl:")
inertia = []
for k in range(2, 8):			 # probiert k von 2 bis 7 durch
    probe = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_skaliert)
    inertia.append(probe.inertia_)	 # inertia_ steht erst nach dem fit zur Verfügung
    print(f"  k = {k}: Inertia = {probe.inertia_:7.1f} | "
          f"Silhouette = {silhouette_score(X_skaliert, probe.labels_):.3f}")  # labels_ enthält je Probe die Gruppennummer
print()

# 4. Modell mit drei Gruppen
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)  # n_init startet zehnmal mit anderen Anfangspunkten und behält das beste Ergebnis
gruppe = kmeans.fit_predict(X_skaliert)	 # trainiert und gibt direkt die Gruppenzuordnung zurück

print("Gruppengroessen:", pd.Series(gruppe).value_counts().sort_index().tolist())  # value_counts zählt, wie oft jede Gruppennummer vorkommt
print()

# 5. Vergleich mit den echten Rebsorten
# Die Gruppennummern sind beliebig vergeben, Gruppe 0 muss nicht Sorte A
# entsprechen. Die Kreuztabelle zeigt die Zuordnung.
vergleich = pd.crosstab(pd.Series(echte_sorte, name="Echte Sorte"),
                        pd.Series(gruppe, name="Gefundene Gruppe"))  # crosstab zählt alle Kombinationen aus Sorte und Gruppe
print("Echte Sorte gegen gefundene Gruppe:")
print(vergleich.to_string())
print()

print("Uebereinstimmung mit den echten Sorten (Adjusted Rand Index):",
      round(adjusted_rand_score(echte_sorte, gruppe), 3))  # vergleicht beide Einteilungen unabhängig von der Nummerierung, 1,0 wäre völlige Übereinstimmung

# 6. Diagramme
plt.figure(figsize=(6, 4))
plt.plot(range(2, 8), inertia, "o-", color="#2c4a7c")  # Inertia über k, gesucht ist der Knick
plt.xlabel("Anzahl Gruppen k")
plt.ylabel("Inertia")
plt.title("Ellenbogenmethode zur Wahl von k")
plt.tight_layout()
plt.savefig(ORDNER / "03_ellenbogen.png", dpi=120)

plt.figure(figsize=(6.5, 5))
for g in range(3):			 # zeichnet jede Gruppe einzeln, damit sie eine eigene Farbe bekommt
    plt.scatter(X.loc[gruppe == g, "Farbtiefe"],
                X.loc[gruppe == g, "Flavonoide"],  # gruppe == g wählt die Zeilen dieser Gruppe aus
                label=f"Gruppe {g}", alpha=0.75, edgecolor="k")
plt.xlabel("Farbtiefe")
plt.ylabel("Flavonoide")
plt.title("k-Means: gefundene Gruppen")
plt.legend()
plt.tight_layout()
plt.savefig(ORDNER / "03_gruppen.png", dpi=120)
print("\nDiagramme gespeichert: 03_ellenbogen.png, 03_gruppen.png")
