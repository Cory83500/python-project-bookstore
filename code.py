import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns 
from numpy import trapz
from scipy.stats import ttest_ind
customers = pd.read_csv("customers.csv")
products = pd.read_csv("products.csv")
transactions = pd.read_csv("transactions.csv")

Rêquetes Antoine. 
# vérifictation des fichiers;

customers.describe()
customers.isnull().sum()

products.describe()
products.drop(products.loc[products['price']< 0].index, inplace=True)
products.describe()
products.isnull().sum()

transactions.describe()
transactions.drop(transactions.loc[transactions['id_prod']== 'T_0'].index, inplace=True)
transactions['date'] = transactions['date'].str[0:10]
transactions.sort_values(["id_prod"]).tail()
transactions.isnull().sum()

# fusion de nos csv;
product_transac = pd.merge(transactions, customers)
data = pd.merge(product_transac, products, how= 'left', on= 'id_prod')
data.dropna(subset= ['price'], inplace= True)
data.shape
# Chiffre d'affaire par client;
CA_customers = data.groupby(['client_id','birth']).sum().reset_index().drop(['categ'], axis= 1).sort_values(by= "price")
CA_customers
Le chiffre d'affaire par client.
# Chiffre d'affaire par client par catégorie;
data.groupby(['client_id','birth','categ']).sum().reset_index().sort_values(by='price')
Le chiffre d'affaire par client par catégorie de livre.
# Chiffre d'affaire total;
CA = data.groupby(['client_id','birth']).sum().drop(['categ'], axis= 1)
CA_total = CA['price'].sum()
print("Le chiffre d'affaire total est de:",round(CA_total/1000000, 2),"M€")
# CA par catégorie;
data.groupby(['categ'])[['price']].sum().reset_index().round(2)
Le chiffre d'affaire par catégorie de livre.
# chiffre d'afffaire par mois; 
CA_date_Mois = data.groupby(['date'])[['price']].sum().reset_index().round(2)
CA_date_Mois['date'] = CA_date_Mois['date'].str[0:7]
CA_date_Mois = CA_date_Mois.groupby(['date'])[['price']].sum().reset_index().round(2)
CA_date_Mois
#creation MM 7 jours;
CA_date_MM = data.groupby(['date'])[['price']].sum().reset_index().round(2)
CA_date_MM['date'] = CA_date_MM['date'].str[0:11]
CA_date_MM = CA_date_MM.groupby(['date'])[['price']].sum().reset_index().round(2)
CA_date_MM['7jours_MM'] = CA_date_MM.price.rolling(7).mean()
CA_date_MM.drop(CA_date_MM.loc[CA_date_MM['date']=='test_2021-0'].index, inplace=True)

#creation du tableau par mois avec sum des MM;
CA_date_MM['date'] = CA_date_MM['date'].str[0:7]
CA_date_MM =CA_date_MM.groupby(['date'])[['price','7jours_MM']].sum().reset_index().round(2)
plt.figure(figsize= (20,7))
plt.plot(CA_date_MM['date'], CA_date_MM['price'], label= "Chiffre d'affaire")
plt.plot(CA_date_MM['date'], CA_date_MM['7jours_MM'], label= "Moyenne mobile 7 jours")
plt.legend()
plt.title("Courbe du CA par mois")
plt.xlabel("Mois")
plt.ylabel("CA")
plt.show()
# regard sur le CA mois d'octobre 
CA_octobre = data.groupby(['date'])[['price']].sum().reset_index().round(2)
CA_octobre.loc[(CA_octobre.date >= '2021-10-01') & (CA_octobre.date <= '2021-10-31')].sort_values(by='price')
# graph moyenne mobile pour les mois de decembre janvier fevrier 2023
CA_date_fin_years = CA_date_MM.loc[CA_date_MM['date']>= '2022-12']
plt.figure(figsize= (20,7))
plt.plot(CA_date_fin_years['date'], CA_date_fin_years['7jours_MM'], label= "Moyenne mobile 7 jours")
plt.legend()
plt.title("Courbe de la moyenne mobile pour la 2022 debut 2023")
plt.xlabel("Mois")
plt.ylabel("CA")
plt.show()

CA_date_fin_years22 = CA_date_MM.loc[(CA_date_MM.date >= '2021-12') & (CA_date_MM.date<= '2022-02')]
plt.figure(figsize= (20,7))
plt.plot(CA_date_fin_years22['date'], CA_date_fin_years22['7jours_MM'], label= "Moyenne mobile 7 jours")
plt.legend()
plt.title("Courbe de la moyenne mobile pour la 2021 debut 2022")
plt.xlabel("Mois")
plt.ylabel("CA")
plt.show()
#zoom sur les références, les tops et les flops;
ref_vente = data.groupby(['id_prod'])[['price']].agg(['count']).reset_index().round(2).sort_values(by=('price', 'count'), ascending=False)
ref_vente
ref_vente_client = data.groupby(['client_id'])[['price']].agg(['sum']).reset_index().round(2).sort_values(by=('price', 'sum'), ascending=False)
ref_vente_client
data.drop(data.loc[data['client_id'] == 'c_6714'].index, inplace=True)
data.drop(data.loc[data['client_id'] == 'c_3454'].index, inplace=True)
data.drop(data.loc[data['client_id'] == 'c_4958'].index, inplace=True)
data.drop(data.loc[data['client_id'] == 'c_1609'].index, inplace=True)
# recalcul du CA global 
CA = data.groupby(['client_id','birth']).sum().drop(['categ'], axis= 1)
CA_total = CA['price'].sum()
print("Le chiffre d'affaire total est de:",round(CA_total/1000000, 2),"M€")
ref_vente = data.groupby(['id_prod','categ'])[['price']].agg(['count']).reset_index().round(2).sort_values(by=('price', 'count'), ascending=False)
ref_vente
References par id_prod 
#répartition top et flot pour catégorie;
ref_vente_categ0 = ref_vente.loc[ref_vente['categ'] == 0.0]
ref_vente_categ1 = ref_vente.loc[ref_vente['categ'] == 1.0]
ref_vente_categ2 = ref_vente.loc[ref_vente['categ'] == 2.0]
quantile0 = ref_vente_categ0['price','count'].quantile([0.25,0.75])
quantile1 = ref_vente_categ1['price','count'].quantile([0.25,0.75])
quantile2 = ref_vente_categ2['price','count'].quantile([0.25,0.75])
plt.figure(figsize=(7,12))
plt.bar(quantile0.index,quantile0.values,width = 0.1, color=(0.1, 0.1, 0.1, 0.1), edgecolor='blue', label= 'catégorie 0')
plt.bar(quantile1.index +0.2,quantile1.values,width = 0.1, color=(0.1, 0.1, 0.1, 0.1), edgecolor='red', label= 'catégorie 1')
plt.bar(quantile2.index+0.1,quantile2.values,width = 0.1, color=(0.1, 0.1, 0.1, 0.1), edgecolor='green', label= 'catégorie 2')
plt.legend()
plt.title("graphique des tops et flops par catégorie")
plt.xlabel("index")
plt.ylabel("top et flop")
plt.show()

Reférence par vente par catégorie 
# CA par age;
data.insert(8, "age", (2023-data['birth']))
age = data.groupby(['age'])[['price']].agg(['sum']).reset_index().round(2)
age
# courbe lorenz;
depenses = age[age['price','sum']> 0]
dep = depenses['price','sum'].values
n = len(dep)
lorenz = np.cumsum(np.sort(dep)) / dep.sum()
lorenz = np.append([0],lorenz) 

plt.figure(figsize= (10,7))
xaxis = np.linspace(0,1,len(lorenz))
plt.plot(xaxis,lorenz, label= 'Courbe de Lorenz')
plt.plot(xaxis, xaxis, label= 'Parfaite répartition')
plt.title('Courbe de Lorenz')
plt.fill_between(xaxis, lorenz, xaxis, color = 'silver', label= 'air A')
plt.fill_between(xaxis, lorenz, color = 'bisque', label= 'air B')
plt.legend()
plt.show()
airB = trapz(lorenz,xaxis)
air = trapz(xaxis,xaxis)
airA = air-airB
coef_gini = airA/(airA + airB)
print("Le coefficient de gini est de:",coef_gini )
Requêtes de Julie. 
# lien entre le genre et la catégorie, Table de contingence et khi 2 (qualitative et qualitative)
cross_tab = pd.crosstab(data.categ, data.sex)
sns.heatmap(cross_tab, annot=True, fmt="g", cmap='viridis')
plt.title("tableau de contingence")
plt.show()
cross_tab = pd.crosstab(data.sex, data.categ, margins= True)
cross_tab
# Test X2;
chi2, p, dof, expected = stats.chi2_contingency(cross_tab)
print("Le chi2 est de:", chi2, "et la p value est de:", p)
On peut voir ici que la p Value est inferieur à 0.05 on considere de ce fait que l'hypothèse H0 n'est pas valide et qu'il y a donc
une dépance et donc qu'il y a bien un lien entre le sex et la catégorie d'achat des consomateur.
#Lien entre l’âge des clients et le montant total des achats (quantitative et quantitative)
fit = np.polyfit(age['age'], age['price','sum'],1)
plt.figure(figsize= (10,7))
plt.scatter(age['age'], age['price','sum'])
plt.title('nuage de point de l age et le prix')
plt.xlabel('age')
plt.ylabel("montant d'achat")
plt.show()

Lien entre l’âge des clients et le montant total des achats, la fréquence d’achat;
#relation age et montant acheté.
r, p = stats.pearsonr(age['age'], age["price","sum"])
print("D'après le test de pearson, le coefficient r est de:", r, "est la p Value est de:", p)
Nous pouvons donc constater que ce résultat de la p Value est en dessous de 0.05, qui est notre seuil de tolérance. Donc par conséquent selon la méthode de Pearson, cela indiquerait qu'il y ait une relation entre nos deux variables. 
# relation entre l'age et la fréquence d'achat; (2 quantitative)
freq = data.groupby(['age'])[['price']].agg(['sum']).reset_index().round(2)
freq["freq_achat"] = freq['price', 'sum']/freq['price', 'sum'].sum()
plt.figure(figsize= (10,7))
plt.scatter(freq['age'], freq['freq_achat'])
plt.title('nuage de point de l age et frequence achats')
plt.xlabel('age')
plt.ylabel("montant d'achat")
plt.show()

# Test pearson;
r, p = stats.pearsonr(freq['age'], freq["freq_achat"])
print("D'après le test de pearson, le coefficient r est de:", r, "est la p Value est de:", p)
Nous pouvons donc constater que ce résultat de la p Value est en dessous de 0.05, qui est notre seuil de tolérance. Donc par conséquent selon la méthode de Pearson, cela indiquerait qu'il y ait une relation entre nos deux variables. 
Lien entre l'age et la taille du panier moyen
#lien entre age et panier moyen; (quantitative et quantitative)
link = data.groupby(['age'])[['price']].agg('mean').reset_index().round(2)
plt.figure(figsize= (10,7))
plt.scatter(link['age'], link['price'])
plt.scatter(link['age'], fit[0]*link['age']+ fit[1])
plt.title('Courbe de régression linaire')
plt.show()

r, p = stats.pearsonr(link['age'], link["price"])
print("D'après le test de pearson, le coefficient r est de:", r, "est la p Value est de:", p)
Nous pouvons donc constater que ce résultat de la p Value est en dessous de 0.05, qui est notre seuil de tolérance. Donc par conséquent selon la méthode de pearson, cela indiquerait qu'il y ait une relation entre nos deux variables. 
# Lien entre l'age et les catégories des livres achetés. (quantitative et qualitative)
plt.figure(figsize= (10,7))
sns.boxplot(x= data.categ, y= data.age)
plt.title("boxplot interval categorie et age")
plt.show()
Lien entre l'age et les catégories des livres achetés.
# Test annova;
stats.f_oneway(data['categ'], data['age'])
Nous pouvons donc constater que ce résultat de la p Value est en dessous de 0.05, qui est notre seuil de tolérance. Donc par conséquent selon la méthode annova, cela indiquerait qu'il y ait une relation entre nos deux variables. 
