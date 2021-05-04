import pandas
import sklearn
import numpy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# instanciation
sc = StandardScaler()
acp = PCA(svd_solver='full')

# data
data = pandas.read_csv("kc_house_data.csv", usecols=[1, 2, 3, 4])
n = data.shape[0]
p = data.shape[1]
print("data : \n", data, "\n")
print("n = ", n, "\n")

# data centrer et réduites
centerReducedData = sc.fit_transform(data)
print("data centrer et réduites : \n", centerReducedData, "\n")

# moyenne (égal à 0)
print("moyenne : \n", numpy.mean(centerReducedData, axis=0), "\n")

# écart-type (égal à 1)
print("écart-type :", numpy.std(centerReducedData, axis=0, ddof=0), "\n")

# calculs des coordonées factorielle
coord = acp.fit_transform(centerReducedData)
print("coordonées factorielle : \n")
print(coord)
# nombre de composantes calculées
print("nombre de composantes calculées", acp.n_components_, "\n")  # 5

# variance expliquée
variance = (n-1)/n*acp.explained_variance_
print("varaince expliquée :", variance, "\n")

# proportion de variance expliquée
print("proportion de variance expliquée: \n",
      acp.explained_variance_ratio_, "\n")

# Variances (Valeurs Propre)
plt.plot(numpy.arange(1, p+1), variance)
plt.title("Variances (Valeurs Propre) par rapport au facteur")
plt.ylabel("Variances (Valeurs Propre)")
plt.xlabel("Facteur")
plt.show()

# cumul de variance expliquée
plt.plot(numpy.arange(1, p+1), numpy.cumsum(acp.explained_variance_ratio_))
plt.title("variance expliquée par rapport au facteur")
plt.ylabel("Addition rapport de variance expliqué")
plt.xlabel("Facteur")
plt.show()

# positionnement des individus dans le plan factoriel
fig, axes = plt.subplots(figsize=(12, 12))
axes.set_xlim(-6, 6)
axes.set_ylim(-6, 6)
for i in range(n):
    plt.annotate(data.index[i], (coord[i, 0], coord[i, 1]))
plt.plot([-6, 6], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
plt.title("positionnement des individus dans le plan factoriel")
plt.show()

# contribution des individus dans l'inertie totale
print("contribution des individus dans l'inertie totale:")
di = numpy.sum(centerReducedData**2, axis=1)
print(pandas.DataFrame({'ID': data.index, 'd_i': di}), "\n")

# contributions aux axes
# permettent de déterminer les individus qui pèsent le plus dans la définition de chaque facteur.
print("contributions aux axes (permettent de déterminer les individus qui pèsent le plus dans la définition de chaque facteur)")
ctr = coord**2
for j in range(p):
    ctr[:, j] = ctr[:, j]/(n*variance[j])
print(pandas.DataFrame(
    {'id': data.index, 'CTR_1': ctr[:, 0], 'CTR_2': ctr[:, 1]}), "\n")
print(numpy.sum(ctr, axis=0), "\n")

# le champ components_(vecteurs propre) de l'objet ACP
print("vecteurs propre: \n", acp.components_, "\n")

# racine carrée des valeurs propres
sqrt_eigval = numpy.sqrt(variance)

# corrélation des variables avec les axes
corvar = numpy.zeros((p, p))
for k in range(p):
    corvar[:, k] = acp.components_[k, :] * sqrt_eigval[k]
# afficher la matrice des corrélations variables x facteurs
print("matrice des corrélations: \n")
print(corvar, "\n")

# on affiche pour les deux premiers axes
print("seulement les deux premiers axes : \n")
print(pandas.DataFrame(
    {'id': data.columns, 'COR_1': corvar[:, 0], 'COR_2': corvar[:, 1]}), "\n")

# cercle des corrélations
fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)
# affichage des étiquettes (noms des variables)
for j in range(p):
    plt.annotate(data.columns[j], (corvar[j, 0], corvar[j, 1]))
# ajouter les axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
# ajouter un cercle
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axes.add_artist(cercle)
plt.title("cercle des corrélations")
plt.ylabel("F2")
plt.xlabel("F1")
# affichage
plt.show()

# cosinus carré des variables
cos2var = corvar**2

# contributions
ctrvar = cos2var
for k in range(p):
    ctrvar[:, k] = ctrvar[:, k]/variance[k]

print("Contribution des variables aux axes pour les deux premiers axes \n")
print(pandas.DataFrame(
    {'id': data.columns, 'CTR_1': ctrvar[:, 0], 'CTR_2': ctrvar[:, 1]}), "\n")


# chargement des individus supplémentaires
indSupp = pandas.read_csv("ajout_individue.csv")
print("nouveaux individus : \n")
print(indSupp, "\n")

# centrage-réduction avec les paramètres des individus actifs
ZIndSupp = sc.transform(indSupp)
print("centrage-réduction avec les paramètres des individus actifs: \n")
print(ZIndSupp, "\n")

# projection dans l'espace factoriel
coordSupp = acp.transform(ZIndSupp)
print("projection dans l'espace factoriel : \n")
print(coordSupp, "\n")

# positionnement des individus supplémentaires dans le premier plan
fig, axes = plt.subplots(figsize=(12, 12))
axes.set_xlim(-6, 6)
axes.set_ylim(-6, 6)
# étiquette des points actifs
for i in range(n):
    plt.annotate(data.index[i], (coord[i, 0], coord[i, 1]))
# étiquette des points supplémentaires (illustratifs) en bleu ‘b’
for i in range(coordSupp.shape[0]):
    plt.annotate(indSupp.index[i], (coordSupp[i, 0],
                 coordSupp[i, 1]), color='b')
# ajouter les axes
plt.plot([-6, 6], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
# affichage
plt.show()

# importation des variables supplémentaires
varSupp = pandas.read_csv("ajout_variables.csv")
print(varSupp)

# variables supplémentaires quantitatives
vsQuanti = varSupp.iloc[:, :2].values
print("variables supplémentaires quantitatives : \n")
print(vsQuanti, "\n")

# corrélation avec les axes factoriels
corSupp = numpy.zeros((vsQuanti.shape[1], p))
for k in range(p):
    for j in range(vsQuanti.shape[1]):
        corSupp[j, k] = numpy.corrcoef(vsQuanti[:, j], coord[:, k])[0, 1]
# affichage des corrélations avec les axes
print("affichage des corrélations avec les axes : \n")
print(corSupp, "\n")

# cercle des corrélations avec les var. supp
fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)
# variables actives
for j in range(p):
    plt.annotate(data.columns[j], (corvar[j, 0], corvar[j, 1]))
# variables illustratives
for j in range(vsQuanti.shape[1]):
    plt.annotate(varSupp.columns[j], (corSupp[j, 0], corSupp[j, 1]), color='g')
# ajouter les axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
# ajouter un cercle
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axes.add_artist(cercle)
plt.title("cercle des corrélations avec 2 variables supplémentaires")
plt.ylabel("F2")
plt.xlabel("F1")
# affichage
plt.show()
