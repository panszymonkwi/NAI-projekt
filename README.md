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
Modele regresji, które są trenowane na tym zbiorze to : model Ridge po
dobraniu optymalnego parametru przy pomocy GridSearchCV, model LinearRegression
oraz model svm.SVR. Wytrenowanie modeli na danych historycznych może posłużyć do
prognozowania cen w czasie rzeczywistym w oparciu o bieżące dane. 
Projekt zawiera liczne diagramy przedstawiające różne zależności cenowe oraz wyniki
testowania wyżej wymienionych modeli.