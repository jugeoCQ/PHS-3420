#!/usr/bin/env python
# coding: utf-8

# ![download.png](attachment:6140fcd4-9004-4a98-9033-b2ce021e84ce.png)

# Le but de ce notebook est de concevoir un circuit quantique capable de tester la violation de l'inégalité de CHSH. La violation de cette inégalité est utilisée pour montrer que la mécanique quantique n'est pas compatible avec les théories locales à variables cachées.

# In[ ]:


import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


# On considère deux observateurs, Alice et Bob, qui partagent une paire de qubits préparée dans un état de Bell maximalement intriqué. Alice a accès au qubit 0 et Bob au qubit 1. Chacun peut choisir entre deux bases de mesure différentes :
# 
# * pour Alice : $A_0$ et $A_1$,
# * pour Bob : $B_0$ et $B_1$.
# 
# À partir des corrélations entre leurs résultats, on construit la quantité suivante :
# 
# $$\langle CHSH \rangle = \langle A_0 B_0 \rangle + \langle A_0 B_1 \rangle + \langle  A_1 B_0 \rangle - \langle A_1 B_1 \rangle .$$
# 
# Une théorie locale à variables cachées impose la borne
# 
# $$\langle CHSH \rangle \leq 2 .$$
# 
# 
# La mécanique quantique prédit cependant qu’il est possible d’atteindre une valeur maximale de $2\sqrt{2}$, en choisissant des bases de mesure adaptées. Nous allons d'abord reproduire numériquement cette violation à l’aide de PennyLane.
# 

# # 1. Simulation

# Pour notre expérience, on prépare la paire de Bell $\vert\phi_{+} \rangle = \frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 11 \rangle)$ avec un circuit quantique. On définit ensuite une fonction générale qui prépare cet état intriqué et mesure la valeur moyenne d’une observable donnée. L’observable à mesurer sera précisée dans les exercices suivants.

# In[ ]:


def bell_pair():
    """Prépare une paire de Bell |Φ+> = (|00> + |11>)/√2 sur les qubits 0 et 1"""
    qml.Hadamard(wires = 0)
    qml.CNOT(wires = [0,1])

dev = qml.device("default.qubit", wires = 2)

#@qml.set_shots(1000) #vous pouvez faire la simulation en mode exact ou avec un nombre de shots fini
@qml.qnode(dev)
def measure_obs(obs):
    """Mesure la valeur moyenne d'une observable donnée sur l'état de Bell"""
    bell_pair()
    return qml.expval(obs)


# ### Exercice : observables d'Alice
# ---
# Il faut maintenant choisir les observables. Alice mesure son qubit (qubit 0)  dans la base $Z$ et base $X$.

# In[ ]:


def alice_observables():
    """
    Complétez les observables de Alice.
    """
    A0 = qml.PauliZ(wires=0)  # votre code ici
    A1 = qml.PauliX(wires=0)  # votre code ici
    return A0, A1


# In[ ]:


A0, A1 = alice_observables()


# ### Exercice : observables de Bob
# ---
# 
# Pour Bob, définissez les mêmes observables que pour Alice ($Z$ et $X$).

# In[ ]:


def bob_observables():
    """
     Complétez les observables de Bob.
    """
    B0 = qml.PauliZ(wires=1)  # votre code ici
    B1 = qml.PauliX(wires=1)  # votre code ici
    
    return B0, B1


# In[ ]:


B0, B1 = bob_observables()


# ### Exercice : Calculer $\langle CHSH \rangle$
# ---
# 
# Complétez la fonction `compute_chsh()` afin qu'elle retourne la valeur d'attente $\langle CHSH \rangle$ . L’inégalité de CHSH est-elle violée pour le choix d'observable défini ci-haut? 

# In[ ]:


def compute_chsh(A0, A1, B0, B1):
    """
    Calcule la valeur de l'opérateur CHSH pour un ensemble d'observables.
    
    Args:
        A0, A1 : observables de Alice
        B0, B1 : observables de Bob
    
    Returns:
        CHSH_expval : valeur de <CHSH>
    """
    observables = [A0 @ B0, A0 @ B1, A1 @ B0, A1 @ B1]

    expvals = [measure_obs(obs) for obs in observables]
 
    # CHSH_expval = <A0 @ B0> + <A0 @ B1> + <A1 @ B0> - <A1 @ B1>
    CHSH_expval = np.sum(expvals[:3]) - expvals[3] # votre code ici
    
    print(' CHSH expectation value: ', CHSH_expval)
    print(' Maximal CHSH violation: ', 2*np.sqrt(2))

    return CHSH_expval


# In[ ]:


chsh = compute_chsh(A0, A1, B0, B1)


# Ce choix ne permet pas de violer l’inégalité de CHSH.
# $$ \langle Z \otimes Z \rangle = 1 $$
# $$ \langle X \otimes X \rangle = 1 $$
#  
# et
# $$ \langle Z \otimes X \rangle = 0 $$ 
# $$ \langle X \otimes Z \rangle = 0 $$.
# 
# Donc $\langle CHSH \rangle = 0$ 

# Un meilleur choix de base pour maximiser la violation de l’inégalité de Bell consiste à mesurer le qubit de Bob dans le plan $Z-X$, c’est-à-dire dans une base tournée par rapport à celles d’Alice.
# 
# ### Exercice : observables optimales de Bob
# ---
# 
# Étant donné $B_0 = \frac{Z + X}{\sqrt{2}}$, déterminez la base $B_1$ permettant d’obtenir la violation maximale de l’inégalité de CHSH, soit $\langle CHSH \rangle = 2\sqrt{2}$. 
# 
# Indice : $B_1$ sera également une combinaison linéaire normalisée de $Z$ et $X$.

# In[ ]:


def bob_observables_optimal():
    """
    Définir les observables de Bob pour atteindre <CHSH> = 2√2.
    """
    B0 = (qml.PauliZ(wires=1) + qml.PauliX(wires=1)) / np.sqrt(2)
    B1 = (qml.PauliZ(wires=1) - qml.PauliX(wires=1)) / np.sqrt(2) #votre code ici

    return B0, B1


# In[ ]:


B0, B1 = bob_observables_optimal()


# ### Exercice : Calculez $\langle CHSH \rangle$
# ---
# Calculez la valeur de $\langle CHSH \rangle$ avec la fonction `compute_chsh`. L’inégalité de CHSH est-elle violée pour ce choix d'observables? 

# In[ ]:


chsh = compute_chsh(A0, A1, B0, B1)


# # 2. Matériel quantique
# 
# Nous allons maintenant réaliser cette expérience sur un processeur quantique réel afin de vérifier si l’inégalité de Bell est bien violée.

# In[ ]:


from pennylane_calculquebec.API.client import CalculQuebecClient
from pennylane_calculquebec.processing.config import  MonarqDefaultConfig, PrintDefaultConfig, NoPlaceNoRouteConfig

from pennylane_calculquebec.processing.steps import IBUReadoutMitigation, MatrixReadoutMitigation, PrintResults


# ### Exercice : Identifiants
# ---
# 
# Complétez le client avec vos identifiants.

# In[ ]:


my_client = CalculQuebecClient(
    host = "https://manager.anyonlabs.com",
    user = "your user",
    access_token = "your access token",
    project_id = "d3d760d4-5eae-4e16-abbc-4170183f61f4", 
)

my_client = CalculQuebecClient(
    host = "https://manager.anyonlabs.com",
    user = "jugeo",
    access_token = "nV52EuuJiujXoAnQsN7XnG2wvmbza0I6",
    project_id = "e9caf1bb-29b4-43f5-aa6a-7e4bb0f2e5f2", 
)


my_config = MonarqDefaultConfig("yukon") # config par défaut 

# Vous pouvez activer la mitigation d'erreur avec :
my_config.steps.append(MatrixReadoutMitigation("yukon")) 


# Nous allons utiliser les bases de mesure qui donne la violation maximale ($2 \sqrt{2}$).
# * pour Alice : $Z$ et $X$,
# * pour Bob : $\frac{Z+ X}{\sqrt{2}}$ et $\frac{Z - X}{\sqrt{2}}$.
# 
# Attention, en raison des contraintes du plugin et du matériel quantique, les valeurs d’attente doivent être mesurées une à la fois. Cela ne change toutefois pas le calcul : la quantité $\langle CHSH \rangle$ est toujours obtenue en combinant les quatre corrélations, exactement comme dans la simulation. Exécuter les cellules ci-dessous.

# In[ ]:


observables = [A0 @ qml.PauliZ(1), A0 @ qml.PauliX(1), A1 @ qml.PauliZ(1), A1 @ qml.PauliX(1)]

dev = qml.device("monarq.backup", client = my_client, wires = 2, processing_config = my_config)
@qml.set_shots(1000)
@qml.qnode(dev)
def measure_obs(obs):
    bell_pair()
    return qml.expval(obs)


# In[ ]:


expvals = [measure_obs(obs) for obs in observables]

A0B0 = (expvals[0] + expvals[1])/np.sqrt(2)
A0B1 = (expvals[0] - expvals[1])/np.sqrt(2)
A1B0 = (expvals[2] + expvals[3])/np.sqrt(2)
A1B1 = (expvals[2] - expvals[3])/np.sqrt(2)

# The CHSH operator is A0 @ B0 + A0 @ B1 + A1 @ B0 - A1 @ B1
CHSH_expval = A0B0 + A0B1 + A1B0 - A1B1
print('CHSH expectation value: ', CHSH_expval)
print('Theoretical value 2sqrt(2) : ', 2*np.sqrt(2))


# 
# Les ordinateurs quantiques actuels sont très sensibles au bruit. Cette expérience illustre que, si le bruit est trop important, nous perdons l’aspect « quantique » du système. Une solution consiste à utiliser des méthodes de **mitigation d’erreur**, par exemple des techniques de post-traitement classiques qui améliorent les résultats.
# 
# Vous pouvez ajouter la ligne suivante lors de la définition de votre configuration pour tester mitigation d'erreurs :
# 
# ```python
# my_config.steps.append(MatrixReadoutMitigation("yukon"))
# ```

# 
# 
# ## Références
# 
# *  [https://quantum.cloud.ibm.com/docs/en/tutorials/chsh-inequality](https://quantum.cloud.ibm.com/docs/en/tutorials/chsh-inequality)
# 
# * [https://pennylane.ai/qml/demos/tutorial_noisy_circuit_optimization](https://pennylane.ai/qml/demos/tutorial_noisy_circuit_optimization)
# 

# In[ ]:




