import pandas
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sc = StandardScaler()

# Data
df = pandas.read_csv("kc_house_data.csv")
dataset = df.values

# Set les matrices X les features et Y notre target
X_entrer = dataset[:, 1:5]
X_entrer = np.append(X_entrer, [[3, 1.5, 1060, 9711]], axis=0)
Y_entrer = dataset[:, 0]

# On divise chaque entré par la valeur max des entrées
X_entrer = X_entrer / np.amax(X_entrer, axis=0)
Y = Y_entrer / np.amax(Y_entrer, axis=0)

# X_scale = sc.fit_transform(X)


# On récupère ce qu'il nous intéresse
# Données sur lesquelles on va s'entrainer
X = np.split(X_entrer, [len(X_entrer)-1])[0]
# Valeur que l'on veut trouver
xPrediction = np.split(X_entrer, [len(X_entrer)-1])[1]


# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.2, random_state=1)
# print(X_train.shape, X_test.shape)


class Neural_Network(object):
    def __init__(self):

        # Nos paramètres
        self.inputSize = 4  # Nombre de neurones d'entrer
        self.outputSize = 1  # Nombre de neurones de sortie
        self.hiddenSize = 5  # Nombre de neurones cachés

        # Nos poids
        # (4x5) Matrice de poids entre les neurones d'entrer et cachés
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        # (5x1) Matrice de poids entre les neurones cachés et sortie
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

        # Fonction de propagation avant
    def forward(self, X):

        # Multiplication matricielle entre les valeurs d'entrer et les poids W1
        self.z = np.dot(X, self.W1)
        # Application de la fonction d'activation (Sigmoid)
        self.z2 = self.sigmoid(self.z)
        # Multiplication matricielle entre les valeurs cachés et les poids W2
        self.z3 = np.dot(self.z2, self.W2)
        # Application de la fonction d'activation, et obtention de notre valeur de sortie final
        o = self.sigmoid(self.z3)
        return o

        # Fonction d'activation
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

        # Dérivée de la fonction d'activation
    def sigmoidPrime(self, s):
        return s * (1 - s)

        # Fonction de rétropropagation
    def backward(self, X, Y, o):
        self.o_error = Y - o.T  # Calcul de l'erreur
        # Application de la dérivée de la sigmoid à cette erreur
        self.o_delta = self.o_error.T*self.sigmoidPrime(o)
        # Calcul de l'erreur de nos neurones cachés
        self.z2_error = self.o_delta.dot(self.W2.T)
        # Application de la dérivée de la sigmoid à cette erreur
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)  # On ajuste nos poids W1
        self.W2 += self.z2.T.dot(self.o_delta)  # On ajuste nos poids W2

        # Fonction d'entrainement
    def train(self, X, Y):

        o = self.forward(X)
        self.backward(X, Y, o)

        # Fonction de prédiction
    def predict(self):

        print("Donnée prédite apres entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))

        # if(self.forward(xPrediction) < 0.5):
        #     print("La fleur est BLEU ! \n")
        # else:
        #     print("La fleur est ROUGE ! \n")


NN = Neural_Network()
# NN.train(X, Y)


for i in range(1000):  # Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    print("# " + str(i) + "\n")
    # print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(Y))
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X), 5)))
    print("\n")
    NN.train(X, Y)

NN.predict()
