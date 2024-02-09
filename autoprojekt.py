import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
print("\n P R O J E K T")
print(" Prognozowanie cen samochodów z udziałem sztucznej inteligencji")
print(" autor : Szymon Kwidzinski")
print("\n 1. Wczytywanie bibliotek")
print("      - import pandas")  
import pandas as pd
#import pandas_profiling   #pip3 install pandas-profiling
print("      - import numpy")
import numpy as np
print("      - import matplotlib")
import matplotlib.pyplot as plt
#import csv
print("      - import seaborn")
import seaborn as sns
print("      - import sys")
import  sys
print("      - import sklearn")
from sklearn import linear_model, svm, preprocessing, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error, r2_score

print(" 2. Wczytanie danych")
print(" 3.Poprawa danych")
print("   3.1 Uzupełnianie niewypełnionych danych")
print("   3.2 Proste skalowanie danych")
print("   3.3 Usuwanie zbędnych kolumn z danymi")
print("   3.4 Wstępna analiza danych  W Y K R E S Y")
print(" 4. PRZETWARZANIE ")
print("   4.1 Wyniki dla modelu GridSearchCV")
print("   4.2 Wyniki dla modelu LinearRegression")
print(" 5. Podsumowanie")

print("\n************ WCZYTANIE DANYCH ************")
csv_filename = 'BazaDanychAut.csv'
try:
   with open(csv_filename, newline='') as csv_file:
      cars = pd.read_csv(csv_file, sep=',')
      csv_file.close()
except FileNotFoundError:
   sys.exit("Uwaga: plik BazaDanychAut.csv nie istnieje!")
print('\nRozmiar zbioru ', cars.shape)
print()
print(cars[:10])
#print(cars.tail())

print("\nTypy parametrów aut")
print(cars.dtypes)

carnames = cars['Nazwa auta'].apply(lambda x: x.split(" ")[0])  #lista pierwszych członow nazw aut do spacji
print("\nLiczebność poszczególnych marek aut")
print(carnames.astype('category').value_counts())  #zliczanie i wypisanie liczby aut wg marki

print("\nUwaga: kontynuacja po zamknieciu diagramu!")
#Diagram liczby samochodów wg marki
carnames.astype('category').value_counts().nlargest(28).plot(kind='bar', figsize=(10,5))
plt.title("Liczba aut wg marki")
plt.ylabel('liczba aut')
plt.xlabel('marka aut')
plt.subplots_adjust(top=0.92, bottom=0.24, left=0.10, right=0.95)
plt.savefig('Figura1.jpeg', dpi=400)
plt.show()

print("\n********** P O P R A W A  D A N Y C H **********")
print("\nUzupełnianie niewypełnionych danych")
# Zliczanie rekordów ze '?' lub pustych w kolumnie 'cena'
n1 = cars['cena'].loc[cars['cena'] == '?'].count()
n2 = cars['cena'].isnull().sum()
print(f"Liczba nieznanych cen = {n1 + n2}")
print('\n', cars[['ID auta','Nazwa auta','cena']].head(10))
print()
if n1 + n2 > 0:
   cars[['cena']] = cars[['cena']].replace('?', np.nan)  # zastępowanie wartości ? przez NaN w kolumnie 'cena'
   cars[['cena']] = cars[['cena']].astype('float64')
   mean = cars['cena'].mean() #liczenie średniej ceny z kolumny 'cena'
   mean = round(mean, 0)
   print(f"Dopisanie brakujących cen - uzupelnianie srednia : {mean}")
   cars[['cena']] = cars[['cena']].replace(np.nan, mean) #zastępowanie wartości NaN przez średnią w kolumnie Cen
   print(cars[['ID auta','Nazwa auta','cena']].head(10))
cars[['cena']] = cars[['cena']].astype('float64')
maxcena = cars['cena'].max()
print(f'maksymalna cena : {maxcena}')
mincena = cars['cena'].min()
print(f'minimalna cena : {mincena}')
mean = cars['cena'].mean()
mean = round(mean, 0)
print(f'Srednia cena : {mean}')

print(f"\nLiczba nieznanych wartości w kolumnach")
cars = cars.replace('?', np.nan)
print(cars.isnull().sum())

#print("\nLiczba aut wg marki")
#carnames = cars['Nazwa auta'].apply(lambda x: x.split(" ")[0])  # lista pierwszych członow nazw aut do spacji
#print(carnames.astype('category').value_counts())

print("\nPoprawa literówek...")
cars['Nazwa auta'] = cars['Nazwa auta'].str.replace('maxda','mazda')
cars['Nazwa auta'] = cars['Nazwa auta'].str.replace('porcshce','porsche')
cars['Nazwa auta'] = cars['Nazwa auta'].str.replace('toyouta','toyota')
cars['Nazwa auta'] = cars['Nazwa auta'].str.replace('vokswagen','volkswagen')
cars['Nazwa auta'] = cars['Nazwa auta'].str.replace('Nissan','nissan')
cars['Nazwa auta'] = cars['Nazwa auta'].str.replace('vw', 'volkswagen')
carnames = cars['Nazwa auta'].apply(lambda x: x.split(" ")[0])  # lista pierwszych członow nazw aut do spacji

print("\nLiczba aut wg marki po poprawie nazw")
print(carnames.astype('category').value_counts())
carnames.astype('category').value_counts().nlargest(22).plot(kind='bar', figsize=(10, 5))
plt.title("Liczba aut wg marki  (po poprawie)")
plt.ylabel('liczba aut')
plt.xlabel('marka aut')
plt.subplots_adjust(top=0.92, bottom=0.24, left=0.10, right=0.95)
plt.savefig('Figura1poprawa.jpeg', dpi=400)
plt.show()
#print(cars['Nazwa auta'].unique())

print("\nZamiana jednostek")
cars[['cena']] = cars[['cena']].astype('float64') #zmiana typu obiekt na float64
cars[['cena']] = 4.04*cars[['cena']] #zmiana dolara na zlotego
cars[['dlugosc']] = 25.4 * cars[['dlugosc']] #zamiana cali na milimetry
cars[['szerokosc']] = 25.4 * cars[['szerokosc']]
cars[['wysokosc']] = 25.4 * cars[['wysokosc']]
cars[['masa']] = 0.454 * cars[['masa']] #zmiana funty na kilogramy
cars[['pojemnosc silnika']] = 16.387 * cars[['pojemnosc silnika']] #zmiana ci(cal**3) na cm**3
cars[['moc silnika']] = 1.34 * 1.36 * cars[['moc silnika']] #zmiana koni mechanicznych HP na KW a nastepnie na KM
cars[['spalanie-miasto']] = 235 / cars[['spalanie-miasto']]  #zamiana mile na galon na litry na 100k
cars[['spalanie-autostrada']] = 235 / cars[['spalanie-autostrada']]
print(cars[['dlugosc','szerokosc','wysokosc','masa','moc silnika','cena','spalanie-miasto']].head())
#cars.rename(columns={'spalanie-miasto':'spalanie-miasto-na100km'},inplace=True)  #zmiana nazwy kolumny
#print(cars.loc[cars['horsepower'] > 200])

print("\nKlasyfikacja cen na: Niska, Średnia, Wysoka")
binwidth = int((max(cars['cena'])- min(cars['cena']))/3)  #dzielimy ceny na 3 równe kategorie
bins = range(int(min(cars['cena'])),int(max(cars['cena'])),binwidth)
cars['poziom ceny']= pd.cut(cars['cena'], bins, labels=["Niska","Średnia","Wysoka"])
print(cars.loc[0:10,['cena', 'poziom ceny', 'Nazwa auta']])

'''
cars_group = cars[['naped', 'typ nadwozia','cena']]
print(cars_group)
print("Usuwanie zbędnych kolumn z danymi")
cars_numeric = cars.drop(['symbol','ID_auta'], axis=1)
#print(cars_numeric.head())
'''

#histogram cen
plt.figure(num=2, figsize=(8,5), dpi=100)
plt.title("Histogram cen aut")
plt.ylabel('liczba aut')
plt.xlabel('cena')
plt.hist(cars['cena'], bins = 20, color='orange')
plt.savefig('Figura2.jpeg', dpi=400)
plt.show()

print("\nWstępna analiza danych  W Y K R E S Y\n")
#wykresy kolumnowe
plt.figure(num=3, figsize=(12,8), dpi=90)
plt.subplot(2, 2, 1)
cars['typ paliwa'].value_counts().plot(kind='bar',color='green')
plt.title("Diagram typów paliwa")
plt.ylabel('liczba aut')
plt.xlabel('typ paliwa')
   
plt.subplot(2, 2, 2)
cars['typ silnika'].value_counts().plot(kind='bar',color='blue')
plt.title("Diagram typów silnika")
plt.ylabel('liczba aut')
plt.xlabel('typ silnika')

#wykres
#plt.figure(num=3, figsize=(9,4), dpi=90)
plt.subplot(2, 2, 3)
sns.countplot(cars['typ nadwozia'])
plt.title('Typ nadwozia')
plt.xlabel('liczba aut')

plt.subplot(2, 2, 4)
sns.countplot(cars['naped'], color="orange")
plt.title('Typ napedu')
plt.xlabel('liczba aut')
plt.subplots_adjust(top=0.97, bottom=0.05, left=0.12, right=0.98)
plt.tight_layout(h_pad=0.3)
plt.savefig('Figura3.jpeg', dpi=400)
plt.show()

#wykresy korelacji
fig = plt.figure(num=4, figsize=(12,9), dpi=80)
fig.suptitle("Punktowo-liniowy wykresy korelacji", fontsize=16)
print("--- Wykres Pojemnosc silnika vs cena ---")
plt.subplot(2, 2, 1)
# wykres punktowo-liniowy (nachylenie linii wskazuje na korelację między „wielkością silnika” a „ceną”.
sns.regplot(x="pojemnosc silnika", y="cena", data= cars)
plt.title("Pojemnosc silnika vs cena")
plt.ylim(0,)
print(cars[["pojemnosc silnika","cena"]].corr())
print('Nachylenie linii wykresu 1 wskazuje na dodatnią korelację między „pojemnością silnika” a „ceną”')

print("--- Wykres Moc silnika vs cena ---")
plt.subplot(2, 2, 2)
# wykres punktowo-liniowy rozrzutu (wraz ze wzrostem mocy silnika rośnie cena)
sns.regplot(x="moc silnika", y="cena", data=cars, color='blue')
plt.title("Moc silnika vs cena")
print(cars[["moc silnika","cena"]].corr())
print('Nachylenie linii wykresu 2 wskazuje na dodatnią korelację między „mocą silnika” a „ceną”')

print("--- Wykres Spalanie-miasto vs cena ---")
plt.subplot(2, 2, 3)
# wykres punktowo-liniowy rozrzutu (wraz ze wzrostem masy własnej auta rośnie cena)
sns.regplot(x="spalanie-miasto", y="cena", data=cars, color = 'red')
plt.title("Spalanie-miasto vs cena")
print(cars[["spalanie-miasto","cena"]].corr())
print('Nachylenie linii wykresu 3 wskazuje na dodatnią korelację między „spalanie-miasto” a „ceną”')

print("--- Wykres Długość auta vs cena ---")
plt.subplot(2, 2, 4)
# wykres punktowo-liniowy rozrzutu (wraz ze wzrostem długości auta rośnie cena)
sns.regplot(x="dlugosc", y="cena", data=cars, color = 'green')
plt.title("Długość auta vs cena")
print(cars[["dlugosc","cena"]].corr())
print('Nachylenie linii wykresu 4 wskazuje na dodatnią korelację między „długością auta” a „ceną”')

plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95)
plt.tight_layout(h_pad=0.4)
plt.savefig('Figura4.jpeg', dpi=400)
plt.show()

''' 
#wykres kolumnowy
plt.figure(num=5, figsize=(17,5), dpi=80)
plt.subplot(1, 3, 1)
sns.boxplot(x="typ nadwozia", y="cena", data=cars)
plt.title("Typ nadwozia vs Cena")
plt.subplot(1, 3, 2)
sns.boxplot(x="liczba drzwi", y="cena", data=cars, color='red')
plt.title("Liczba drzwi vs Cena")
plt.subplot(1, 3, 3)
sns.boxplot(x="liczba cylindrow", y="cena", data=cars, color='violet')
plt.title("Liczba cylindrow vs Cena")
plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.97)
#plt.show()
'''

print("\n******************* PRZETWARZANIE *******************")
# podzial cars na X i y
X = cars.loc[:, ['symbol', 'typ paliwa', 'turbo/std', 'liczba drzwi',
       'typ nadwozia', 'naped', 'dlugosc', 'szerokosc', 'wysokosc', 'masa',
       'typ silnika', 'liczba cylindrow', 'pojemnosc silnika', 'moc silnika',
       'spalanie-miasto', 'spalanie-autostrada'
       ]]

y = cars['cena']

# przeksztalcanie zbioru X
cars_categorical = X.select_dtypes(include=['object'])   #lista kolumn z elementami typu object
print("\nKolumny z wartościami typu object")
print(cars_categorical.head())
#Konwertowanie zmiennych kategoryczne w zmienne manekinowe
print("\nTworzenie zamiennych kolumn")
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True, dtype=float) #tworzenie nowych kolumn z wartosciami 0 lub 1
print(cars_dummies.head())
print("Usuwanie kolumn z wartosciami katerogytycznymi")
X = X.drop(list(cars_categorical.columns), axis=1) #usuwanie kolumn z wartosciami kategorycznymi
print(X.head())
print("Dolaczenie nowo utworzonych kolumn")
X = pd.concat([X, cars_dummies], axis=1) #dolaczenie nowo utworzonych kolumn
print(X.head())

# podzial na train i test
print("\nPodział danych na treningowe i testowe oraz skalowanie danych")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=100)

#skalowanie danych
scaler = StandardScaler()
scale_features = ['liczba drzwi', 'dlugosc', 'szerokosc', 'wysokosc', 'masa', 
       'pojemnosc silnika', 'moc silnika', 'spalanie-miasto', 'spalanie-autostrada']

#print("Kolumny z wartosciami")
#print(scale_features)
print("\nZbiory treningowy X_train")
X_train[scale_features]=scaler.fit_transform(X_train[scale_features])
print(X_train.head())
print("Rozmiar X_train ", X_train.shape)
print("\nZbiór testowy X_test")
X_test[scale_features]=scaler.fit_transform(X_test[scale_features])
print(X_test.head())
print("Rozmiar X_test ", X_test.shape)
print("\nCeny rzeczywiste zbioru testowgo y_test")
print(y_test.head())
print("Rozmiar y_test ", y_test.shape)

# przetwarzanie
print("\n   W Y N I K I  model Ridge (GridSearchCV)")
print("\nWyszukiwanie przez GridSearchCV najlepszych wartości parametrów dla estymatora Ridge.")
params={'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3,
                                   0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50,
                                   100, 500, 1000]}
model_cv = GridSearchCV(estimator = Ridge(), 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = 5, 
                        return_train_score=True,
                        verbose = 0)
model_cv.fit(X_train, y_train)
#cv_results = pd.DataFrame(model_cv.cv_results_)
#print(cv_results.head())
print("Najlepszy parametr: ", model_cv.best_params_)
alfa = model_cv.best_params_["alpha"]
#najlepszy model
ridge = Ridge(alpha=alfa)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
df= pd.DataFrame({'cena':y_test,'cena_pred':y_pred})
df['cena']= round(df['cena'], 2)
df['cena_pred']= round(df['cena_pred'], 2)
df['Nazwa auta'] = cars['Nazwa auta']
print(df.head(10))
maxcena = cars['cena'].max()
mincena = cars['cena'].min()
wsp = maxcena - mincena
NRMSE = np.sqrt(mean_squared_error(y_test, y_pred))/wsp #RMSE(pierwiastek kwadratowy błędu średniokwadratowego)
R2 = r2_score(y_test, y_pred)                           #NRMSE znormalizowany RMSE(podzielony przez maxcena-mincena)
print()
#ocena modelu
print('ocena modelu ', Ridge(alpha=alfa))
print('NRMSE:',NRMSE, '    R-Squared:',R2)

plt.figure(num=5, figsize=(14,8), dpi=80)
plt.subplot(2, 3, 1)
sns.regplot(x = y_test, y = y_pred, color='red')
plt.scatter(y_test,y_pred)
plt.title('Cena vs Prognoza - model Ridge')
plt.xlabel('cena', fontsize=10)
plt.xticks([0, 40000, 80000, 120000, 160000, 200000])
plt.ylabel('prognozowana cena', fontsize=10)
plt.ylim(0,180000)

indeks = [i for i in range(1, y_pred.size + 1, 1)]  # generating index
plt.subplot(2, 3, 4)
plt.plot(indeks, y_test, color="red", linewidth=2, linestyle="-")
plt.plot(indeks, y_pred, linewidth=2, linestyle="-")
plt.title('Wykresy cen')
plt.xlabel('indeks', fontsize=10)
plt.ylabel('cena', fontsize=10)
plt.legend(["cena","cena_pred"], loc=1, borderpad=0.15)
plt.ylim(0,200000)

print("\n   W Y N I K I  model LinearRegression")
#from sklearn.cross_validation import train_test_split

linearmodel = LinearRegression()
linearmodel.fit(X_train,y_train)
score = linearmodel.score(X_test, y_test) #R-squared
y_pred = linearmodel.predict(X_test)

df = pd.DataFrame()
df['cena'] = y_test
df['cena']= round(df['cena'],2)
df['cena-pred'] = y_pred
df['cena-pred']= round(df['cena-pred'],2)
df['Nazwa auta'] = cars['Nazwa auta']
print(df.head(10))
maxcena = cars['cena'].max()
mincena = cars['cena'].min()
wsp = maxcena - mincena
NRMSE = np.sqrt(mean_squared_error(y_test, y_pred))/wsp
R2 = r2_score(y_test, y_pred)
print()
#ocena modelu
print('ocena modelu', LinearRegression())
print('NRMSE:',NRMSE, '    R-Squared:',R2)

plt.subplot(2, 3, 2)
sns.regplot(x = y_test, y = y_pred, color='red')
plt.scatter(y_test, y_pred, color='green')
plt.title('Cena vs Prognoza - model LinearRegression')
plt.xlabel('cena', fontsize=10)
plt.xticks([0, 40000, 80000, 120000, 160000, 200000])
plt.ylim(0, 180000)

indeks = [i for i in range(1, y_pred.size + 1, 1)]  # generating index
plt.subplot(2, 3, 5)
plt.plot(indeks, y_test, color="red", linewidth=2, linestyle="-")
plt.plot(indeks, y_pred, color="green", linewidth=2, linestyle="-")
plt.title('Wykresy cen')
plt.xlabel('indeks', fontsize=10)
plt.legend(["cena","cena_pred"], loc=1, borderpad=0.15)
plt.ylim(0,200000)

print("\n   W Y N I K I  model svm.SVR")
clf = svm.SVR(kernel="linear")
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test) #R-squared
y_pred = clf.predict(X_test)

df = pd.DataFrame()
df['cena'] = y_test
df['cena']= round(df['cena'],2)
df['cena-pred'] = y_pred
df['cena-pred']= round(df['cena-pred'],2)
df['Nazwa auta'] = cars['Nazwa auta']
print(df.head(10))
maxcena = cars['cena'].max()
mincena = cars['cena'].min()
wsp = maxcena - mincena
NRMSE = mean_squared_error(y_test, y_pred)/wsp
print()
#ocena modelu
print('ocena modelu', svm.SVR(kernel='linear'))
print('NRMSE:',NRMSE, '    R-Squared:',R2)

plt.subplot(2, 3, 3)
sns.regplot(x = y_test, y = y_pred, color='red')
plt.scatter(y_test,y_pred, color='blue')
plt.title('Cena vs Prognoza - model svm.SVR')
plt.xlabel('cena', fontsize=10)
plt.xticks([0, 40000, 80000, 120000, 160000, 200000])
plt.ylim(0, 180000)

indeks = [i for i in range(1, y_pred.size + 1, 1)]  # generating index
plt.subplot(2, 3, 6)
plt.plot(indeks, y_test, color="red", linewidth=2, linestyle="-")
plt.plot(indeks, y_pred, color="blue", linewidth=2, linestyle="-")
plt.title('Wykresy cen')
plt.xlabel('indeks', fontsize=10)
plt.ylim(0,200000)
plt.subplots_adjust(top=0.96, bottom=0.06, left=0.07, right=0.99)
plt.tight_layout(h_pad=0.4)
plt.legend(["cena","cena_pred"], loc=1, borderpad=0.15)
plt.savefig('Figura5.jpeg', dpi=400)
plt.show()

print('\n******************** PODSUMOWANIE ********************')
print("Dwa pierwsze modele uzyskały dobre wyniki na danych historycznych")
print("i mogą posłużyć do prognozowania cen w czasie rzeczystym w oparciu")
print("o bieżące dane samochodów. W obu przypadkach wynik NRMSE jest bliski 0,")
print("a wynik dopasowania (wspolczynnik determinacji) R-Squared jest bliski 1.")
print("Wynik dopasowania w przypadku trzeciego modelu jest ujemny, co swiadczy")
print("o zlym dopasowaniu modelu. Porownujac modele najlepszy wynik otrzymal LinearalRegresion.")

print("\nUwaga: zakończenie programu po zamknięciu diagramu!")
plt.show()