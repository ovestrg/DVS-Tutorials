# DVS-Tutorials – Wertstoffhandel

**Modul:** Datenverarbeitungssysteme (DVS)  
**Seminargruppe:** 3IT24-1  
**Projektmitglieder:** Ove Flatow und Linh Thu Nguyen  
**Hochschule:** Duale Hochschule Sachsen

## 1. Projektüberblick

In diesem Repository befinden sich unsere vier Tutorials aus dem Modul Datenverarbeitungssysteme. Als gemeinsames Anwendungsszenario verwenden wir einen Wertstoffhandel mit zwei Annahmestellen und einem Verwertungswerk.

Die Tutorials behandeln unterschiedliche Möglichkeiten, Daten zu speichern, zusammenzuführen und auszuwerten. Dabei bauen nicht alle Teile technisch direkt aufeinander auf.

Das Data Warehouse aus Tutorial 1 bildet die Grundlage für das Reporting in Tutorial 2. In Tutorial 3 untersuchen wir MongoDB als zusätzliche Möglichkeit zur Speicherung von Maschinendaten. Tutorial 4 behandelt verschiedene Machine-Learning-Verfahren anhand von Beispieldatensätzen.

## 2. Reihenfolge und Zusammenhang der Tutorials

| Tutorial | Inhalt | Zusammenhang |
|---|---|---|
| **1 – Data Warehouse** | Aufbau eines Data Warehouses mit PostgreSQL | Grundlage für Tutorial 2 |
| **2 – Reporting** | Erstellung eines Dashboards mit Power BI | Verwendet Daten aus Tutorial 1 |
| **3 – NoSQL** | Speicherung und Auswertung von Maschinendaten mit MongoDB | Ergänzt das Szenario um eine weitere Datenquelle |
| **4 – Machine Learning** | Supervised und Unsupervised Learning mit Python | Eigenständige Beispiele mit Bezug zum Data Warehouse |

**Empfohlene Reihenfolge:** Zuerst Tutorial 1, anschließend Tutorial 2. Tutorial 3 und 4 können danach unabhängig voneinander betrachtet werden.

## 3. Tutorial 1 – Data Warehouse

**Ordner:** [Tut001 – Data Warehouse](./Tut001%20-%20dwh/)

Im ersten Tutorial haben wir zwei getrennte Quellsysteme für die Annahme und Verwertung von Altmetall aufgebaut. Die Daten werden mit PostgreSQL in einem Data Warehouse zusammengeführt.

Das Data Warehouse besteht aus den Schichten Staging, Core und Business. In der Business-Schicht stehen zwei Faktentabellen für den Ankauf und die Verwertung zur Verfügung.

Damit untersuchen wir unter anderem, welche Mengen an den beiden Standorten angekauft werden und wie hoch die Ausbeute nach der Sortierung ist.

**Wichtige Dateien:**
- `README_DWH.md` – Beschreibung des Data Warehouses
- `Wertstoffhandel/01-Quellsysteme/` – SQL-Dateien für die beiden Quellsysteme
- `Wertstoffhandel/02-dwh/` – Aufbau und Beladung des Data Warehouses
- `Wertstoffhandel/03-Auswertung/` – SQL-Auswertungen und Screenshots
- `Wertstoffhandel/04-Modelle/` – ER-Modelle und Star-Schema

**Zum Nachvollziehen:** Zuerst die README des Tutorials lesen. Anschließend die Quellsysteme und das Data Warehouse anhand der SQL-Dateien aufbauen. Die Auswertungen befinden sich im Ordner `03-Auswertung`.

## 4. Tutorial 2 – Reporting

**Ordner:** `Tut002 - reporting/`

Im zweiten Tutorial verwenden wir die Daten aus der Business-Schicht des Data Warehouses für ein Dashboard in Power BI.

Das Dashboard zeigt unter anderem Ankaufswerte, Mengen und Ausbeutequoten. Außerdem lassen sich die Ergebnisse nach Standorten, Materialien und Zeiträumen untersuchen.

**Wichtige Dateien:**
- `tutorial002_reporting.pbix` – Power-BI-Dashboard
- `Reporting_Tutorial_Dashboard_Wertstoffhandel.docx` – Dokumentation mit Aufbau, Funktionen und Ergebnissen

**Zum Nachvollziehen:** Die Word-Dokumentation erklärt das Dashboard und die verwendeten OLAP-Funktionen. Die `.pbix`-Datei kann mit Power BI Desktop geöffnet werden.

Die verwendeten Daten stammen aus Tutorial 1. Die Verbindung zwischen den beiden Tutorials besteht also direkt über das Data Warehouse.

## 5. Tutorial 3 – NoSQL

**Ordner:** `Tut003 - NoSQL/`

Im dritten Tutorial untersuchen wir MongoDB anhand von Maschinendaten der Sortieranlagen.

Anders als bei den relationalen Quellsystemen können die Dokumente unterschiedliche Messwerte enthalten. Im Tutorial werden unter anderem das Einfügen, Suchen, Ändern, Löschen und Aggregieren von Dokumenten gezeigt.

Außerdem wird beschrieben, wie sich die Maschinendaten über die Chargennummer mit den Daten des Wertstoffhandels verbinden lassen.

**Wichtige Dateien:**
- `NoSQL_Tutorial_MongoDB.docx` – Beschreibung der Umsetzung und der MongoDB-Befehle
- PNG-Dateien im selben Ordner – Screenshots der einzelnen Arbeitsschritte und Ergebnisse

**Zum Nachvollziehen:** Die Schritte und Befehle sind in der Word-Dokumentation beschrieben. Die Screenshots zeigen die Durchführung.

Dieses Tutorial ergänzt das gemeinsame Szenario um Maschinendaten. Es ist jedoch nicht Bestandteil der automatischen ETL-Beladung aus Tutorial 1.

## 6. Tutorial 4 – Machine Learning

**Ordner:** `Tut004 - MachineLearning/`

Im vierten Tutorial haben wir vier Python-Skripte erstellt und damit insgesamt fünf Verfahren untersucht:

- Lineare Regression
- Decision Tree
- Random Forest
- k-Means
- Hauptkomponentenanalyse (PCA)

Für die Versuche verwenden wir die Diabetes- und Wine-Datensätze aus scikit-learn. Die Python-Skripte sind in Supervised und Unsupervised Learning unterteilt.

**Wichtige Dateien:**
- `ML_Tutorial.docx` – Dokumentation der Verfahren und Ergebnisse
- `Supervised Learning/` – lineare Regression, Decision Tree und Random Forest
- `Unsupervised Learning/` – k-Means und PCA

In den jeweiligen Unterordnern liegen die Python-Skripte und die erzeugten Diagramme.

**Zum Nachvollziehen:** Die Word-Dokumentation beschreibt die Verfahren und die Ergebnisse. Die einzelnen Skripte können mit Python und den benötigten Bibliotheken ausgeführt werden.

Die Machine-Learning-Versuche verwenden keine Daten aus unserem Data Warehouse. Stattdessen erklären wir in der Dokumentation, wie die Verfahren auf Daten aus einem Data Warehouse angewendet werden könnten.

## 7. Hinweise zur Durchführung

Für die vier Tutorials werden unterschiedliche Programme benötigt:

| Tutorial | Benötigte Software |
|---|---|
| 1 | PostgreSQL |
| 2 | Power BI Desktop |
| 3 | MongoDB und MongoDB Shell (`mongosh`) |
| 4 | Python mit scikit-learn, pandas und matplotlib |

Die einzelnen Tutorials enthalten jeweils ihre eigene Dokumentation. Die SQL-Dateien, das Power-BI-Dashboard, die MongoDB-Screenshots und die Python-Skripte sind in den entsprechenden Ordnern abgelegt.

## 8. Zusammenfassung

Die vier Tutorials zeigen verschiedene Möglichkeiten der Datenverarbeitung am Beispiel eines Wertstoffhandels.

Zunächst werden die Daten aus zwei Quellsystemen in einem Data Warehouse zusammengeführt. Anschließend werden sie mit Power BI ausgewertet. MongoDB zeigt eine weitere Möglichkeit, zusätzliche Maschinendaten zu speichern und auszuwerten. Im letzten Tutorial werden verschiedene Machine-Learning-Verfahren ausprobiert und mögliche Anwendungen auf Data-Warehouse-Daten beschrieben.

So entsteht ein gemeinsamer fachlicher Zusammenhang, auch wenn nicht alle Tutorials technisch direkt aufeinander aufbauen.
