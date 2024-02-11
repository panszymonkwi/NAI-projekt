# Projekt na zaliczenie przedmiotu Narzędzia sztucznej inteligencji


### Tytuł: Prognozowanie cen samochodów z udziałem sztucznej inteligencji.
### Autor: Szymon Kwidzinski

## Cel projektu :
Znalezienie odpowiedniego modelu do prognozowania cen 
samochodów w oparciu o zebrane dane o pojazdach. W dzisiejszych czasach kupno
odpowiedniego auta za rozsądną cenę jest ważnym problemem większości
młodych ludzi i nie tylko, a ponieważ też jestem tym zainteresowany,
to zagadnienie wydało mi się bardzo ciekawe. Sztuczna inteligencja
wielkimi krokami weszła w nasz świat, zatem można ją wykorzystać do
rozwiązywania różnych problemów w różnych dziedzinach także do celów
prognozy. Dynamiczne ustalania cen optymalnych wspierane
metodami AL ma na celu pomoc firmom w prognozowaniu popytu a indywidualnym
użytkownikom pomóc w podjęciu decyzji przy zakupie samochodu.

## Opis projektu :
Do prognozowania cen utworzyłem zbiór BazaDanychAut.csv, który otrzymałem
ze zbioru CarPrice_Assignment.csv samochodów amerykańskich z przed ponad 30 lat po
wstępnej obróbce (usunięciu kilku kolumn, które uznałem za nieistotne przy
kupnie samochodu oraz zmian nazw kolumn na język polski).
Modele regresji, które są trenowane na tym zbiorze to: model Ridge po
dobraniu optymalnego parametru przy pomocy GridSearchCV, model LinearRegression
oraz model svm.SVR. Wytrenowanie modeli na danych historycznych może posłużyć do
prognozowania cen w czasie rzeczywistym w oparciu o bieżące dane. 
Projekt zawiera liczne diagramy przedstawiające różne zależności cenowe oraz wyniki
testowania wyżej wymienionych modeli.

## Język kodu :
Python 3.11

## Biblioteki :
(również zawarte w pliku requirements.txt)
* warnings
* pandas
* numpy
* matplotlib
* seaborn
* sys
* sklearn
* liczne podbiblioteki sklearn

## Dodatkowe pliki :
BazaDanychAut.csv

## Uruchomienie :
Po ściągnięciu kodu oraz pliku BazaDanychAut.csv i załadowaniu powyższych bibliotek, wystarczy
uruchomić kod w środowisku python. Kod będzie zatrzymywał się na kolejnych
diagramach i wznawiał działanie po zamknięciu diagramu. Konsola będzie pokazywała informacje 
na temat danego diagramu na podstawie pliku csv oraz algorytmów i kodu. Każdy diagram będzie 
zapisywany w miejscu, z którego został uruchomiony kod.

## Źródła i inspiracje : 
- materiały z przedmiotów : NAJ, IML, FDL, ICK
- [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [GridSearchCV for Beginners](https://towardsdatascience.com/gridsearchcv-for-beginners-db48a90114ee)
- [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Linear Regression in Python](https://realpython.com/linear-regression-in-python/)
- [sklearn.svm.SVR — scikit-learn 1.4.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [PROSTO O DOPASOWANIU PROSTYCH, CZYLI ANALIZA REGRESJI LINIOWEJ W PRAKTYCE, Janusz Wątroba, StatSoft Polska Sp. z o.o.](https://media.statsoft.pl/_old_dnn/downloads/analiza_regresji_liniowej_w_praktyce.pdf) 
- [Zarządzanie cenami za pomocą modeli sztucznej inteligencji](https://webwizard.com.pl/pl/dynamiczne-ustalanie-cen-za-pomoc%C4%85-algorytm%C3%B3w-sztucznej-inteligencji.html)
- Teoria sterowania. Grafika w środowisku MATLIB PG M.Grochowski, R.Piotrowski, Ł.Michalczyk 
- [matplotlib.pyplot — Matplotlib 3.5.3 documentation](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) 
- [Kaggle Wyzwanie - Titanic](https://alexiej.github.io/kaggle-titanic/)
- Linear-Regression/CarPrice_Assignment.csv (korzystanie z wielu wyników, które pokazał Google)

