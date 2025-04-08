#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import time
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from pyswarm import pso


# numpy : bibliothèque pour les calculs numériques et les tableaux multidimensionnels.
# 
# pandas : permet de manipuler facilement des tableaux de données (DataFrames).
# 
# matplotlib.pyplot : pour créer des graphiques comme des courbes ou histogrammes.
# 
# seaborn : bibliothèque de visualisation basée sur matplotlib, plus esthétique et orientée statistiques.
# 
# datasets (de sklearn) : contient des jeux de données standards (comme iris, digits, etc.).
# 
# SVC : classifieur SVM (Support Vector Classifier), utile pour des tâches de classification.
# 
# train_test_split : divise les données en deux ensembles (entraînement et test).
# 
# cross_val_score : permet d’évaluer un modèle avec validation croisée (plus fiable).
# 
# accuracy_score : mesure la précision globale des prédictions.
# 
# recall_score : mesure la capacité du modèle à retrouver les exemples positifs.
# 
# f1_score : moyenne harmonique entre la précision et le rappel, utile pour les classes déséquilibrées.
# 
# time : mesure le temps d’exécution (utile pour comparer la vitesse des algorithmes).
# 
# tqdm : affiche une barre de progression lors de longues boucles ou simulations.
# 
# GridSearchCV : teste toutes les combinaisons possibles d’hyperparamètres (exhaustif mais lent).
# 
# RandomizedSearchCV : teste un échantillon aléatoire de combinaisons (plus rapide).
# 
# BayesSearchCV : optimisation bayésienne des hyperparamètres, plus intelligente et efficace.
# 
# pso : implémente l’algorithme PSO (Particle Swarm Optimization), une technique méta-heuristique inspirée des comportements d'essaims (oiseaux, poissons).

# In[2]:


# Configuration des paramètres
NP = 50          # Taille de la population
F = 0.8          # Facteur de mutation
CR = 0.9         # Taux de recombinaison
GENERATIONS = 100 # Critère d'arrêt
BOUNDS = {
    'C': (1e-3, 1e3),    # Échelle logarithmique
    'gamma': (1e-5, 1e1) # Échelle logarithmique
}


# NP : taille de la population, c'est le nombre de solutions candidates (individus) dans l'algorithme d'Évolution Différentielle.
# 
# F : facteur de mutation, utilisé pour perturber les solutions existantes et explorer de nouveaux espaces de recherche.
# 
# CR : taux de recombinaison, détermine la probabilité que deux solutions s'échangent des informations lors de l'étape de reproduction.
# 
# GENERATIONS : nombre de générations (itérations) que l'algorithme va effectuer avant de s'arrêter.
# 
# BOUNDS : détermine les bornes (plages) des hyperparamètres à optimiser (ici, C et gamma de SVM) sur une échelle logarithmique.

# In[3]:


# Chargement du jeu de données "breast_cancer" de sklearn
data = datasets.load_breast_cancer()  # Chargement des données sur le cancer du sein

# Séparation des données et des étiquettes (features et target)
X, y = data.data, data.target

# Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[4]:


# Fonction d'évaluation qui calcule la précision du modèle SVM via validation croisée
def evaluate(individual):
    C, gamma = individual  # Extraction des valeurs des paramètres C et gamma
    
    # Transformation des paramètres en échelle normale (10^x)
    C_val = 10**C
    gamma_val = 10**gamma
    
    # Création du modèle SVM avec les paramètres transformés
    model = SVC(C=C_val, gamma=gamma_val, random_state=42)
    
    # Validation croisée : évaluation de la précision sur 5 plis
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Retourne la moyenne des scores (précision)
    return np.mean(scores)


# In[5]:


# Initialisation de la population (en échelle log)
def initialize_population():
    # Création d'un tableau de zéros de forme (NP, 2) pour stocker les individus
    population = np.zeros((NP, 2))
    
    # Initialisation des paramètres C et gamma dans une plage logarithmique
    population[:, 0] = np.random.uniform(np.log10(BOUNDS['C'][0]), np.log10(BOUNDS['C'][1]), NP)  # Paramètre C
    population[:, 1] = np.random.uniform(np.log10(BOUNDS['gamma'][0]), np.log10(BOUNDS['gamma'][1]), NP)  # Paramètre gamma
    
    return population


# In[6]:


# Fonction pour maintenir les individus dans les limites des bornes
def ensure_bounds(vec):
    # Limiter la valeur de C dans l'intervalle des bornes logarithmiques
    vec[0] = np.clip(vec[0], np.log10(BOUNDS['C'][0]), np.log10(BOUNDS['C'][1]))
    
    # Limiter la valeur de gamma dans l'intervalle des bornes logarithmiques
    vec[1] = np.clip(vec[1], np.log10(BOUNDS['gamma'][0]), np.log10(BOUNDS['gamma'][1]))
    
    return vec


# In[7]:


# Algorithme d'Évolution Différentielle (DE)
def differential_evolution():
    # Initialisation de la population
    population = initialize_population()
    
    # Évaluation initiale de la population
    fitness = np.array([evaluate(ind) for ind in population])
    
    # Meilleur individu et sa fitness
    best_idx = np.argmax(fitness)
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]
    
    # Historique des performances
    fitness_history = [best_fitness]
    
    # Boucle principale (itérations de l'algorithme)
    for gen in tqdm(range(GENERATIONS), desc="Optimisation"):
        for i in range(NP):
            # Sélection de 3 individus distincts
            candidates = [idx for idx in range(NP) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            
            # Mutation : création d'un mutant
            mutant = a + F * (b - c)
            mutant = ensure_bounds(mutant)  # Appliquer les limites aux paramètres du mutant
            
            # Croisement : création d'un individu d'essai
            cross_points = np.random.rand(2) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, 2)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial = ensure_bounds(trial)  # Appliquer les limites à l'individu d'essai
            
            # Évaluation de l'individu d'essai
            trial_fitness = evaluate(trial)
            
            # Sélection : remplacer si le nouvel individu est meilleur
            if trial_fitness > fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # Mise à jour du meilleur individu
                if trial_fitness > best_fitness:
                    best_individual = trial
                    best_fitness = trial_fitness
        
        # Enregistrement de la meilleure fitness à chaque génération
        fitness_history.append(best_fitness)
    
    return best_individual, fitness_history, population


# In[8]:


# Fonction pour évaluer les performances finales du modèle SVM
def evaluate_performance(C, gamma):
    # Transformation des paramètres
    C_val = 10**C  # Transformation logarithmique de C
    gamma_val = 10**gamma  # Transformation logarithmique de gamma
    
    # Création et entraînement du modèle SVM
    model = SVC(C=C_val, gamma=gamma_val, random_state=42)
    model.fit(X_train, y_train)  # Entraînement sur l'ensemble d'entraînement
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Mesure du temps de prédiction pour 1 observation
    start_time = time.time()
    for _ in range(100):  # Faire 100 prédictions pour évaluer le temps
        model.predict(X_test[:1])  # Prédire sur une seule instance
    pred_time = (time.time() - start_time) * 10  # Temps pour une prédiction en millisecondes
    
    # Calcul des métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Retour des résultats sous forme de dictionnaire
    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1,
        'pred_time': pred_time
    }


# In[9]:


# Fonction pour optimiser les hyperparamètres avec Grid Search
def grid_search():
    # Définition de la grille des hyperparamètres à tester
    param_grid = {
        'C': np.logspace(-3, 3, 7),  # Plage logarithmique pour C
        'gamma': np.logspace(-5, 1, 7)  # Plage logarithmique pour gamma
    }
    
    # Initialisation du GridSearchCV avec un modèle SVM et validation croisée (5 plis)
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
    
    # Entraînement du modèle sur les données d'entraînement
    grid.fit(X_train, y_train)
    
    # Retourner les meilleurs paramètres et le meilleur score
    return grid.best_params_, grid.best_score_


# In[10]:


# Fonction pour optimiser les hyperparamètres avec Randomized Search
def random_search():
    # Définition de la distribution des hyperparamètres à tester
    param_dist = {
        'C': np.logspace(-3, 3, 1000),  # Plage logarithmique pour C
        'gamma': np.logspace(-5, 1, 1000)  # Plage logarithmique pour gamma
    }
    
    # Initialisation du RandomizedSearchCV avec un modèle SVM et validation croisée (5 plis)
    random = RandomizedSearchCV(SVC(random_state=42), param_dist, n_iter=50, cv=5, scoring='accuracy')
    
    # Entraînement du modèle sur les données d'entraînement
    random.fit(X_train, y_train)
    
    # Retourner les meilleurs paramètres et le meilleur score
    return random.best_params_, random.best_score_


# In[11]:


# Fonction pour optimiser les hyperparamètres avec Bayesian Optimization
def bayesian_optimization():
    # Initialisation de l'optimiseur bayésien avec un modèle SVM et validation croisée (5 plis)
    opt = BayesSearchCV(
        SVC(random_state=42),  # Modèle SVM
        {
            'C': (1e-3, 1e3, 'log-uniform'),  # Intervalle pour C, distribution logarithmique
            'gamma': (1e-5, 1e1, 'log-uniform')  # Intervalle pour gamma, distribution logarithmique
        },
        n_iter=50,  # Nombre d'itérations
        cv=5,  # Validation croisée à 5 plis
        scoring='accuracy'  # Critère de scoring : précision
    )
    
    # Entraînement du modèle sur les données d'entraînement
    opt.fit(X_train, y_train)
    
    # Retourner les meilleurs paramètres et le meilleur score
    return opt.best_params_, opt.best_score_


# In[12]:


# Fonction pour optimiser les hyperparamètres avec Particle Swarm Optimization (PSO)
def pso_optimization():
    # Fonction objectif à minimiser : évaluation du modèle SVM
    def objective_function(params):
        # Extraction des paramètres C et gamma
        C, gamma = params
        # Transformation des paramètres logarithmiques en valeurs normales
        C_val = 10**C
        gamma_val = 10**gamma
        
        # Création et validation croisée du modèle SVM
        model = SVC(C=C_val, gamma=gamma_val, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # La fonction PSO minimise la fonction objectif, donc on inverse les scores
        return -np.mean(scores)  # Nous retournons le score négatif pour maximiser la performance
    
    # Définition des limites pour les paramètres C et gamma (logarithmiques)
    lb = [np.log10(BOUNDS['C'][0]), np.log10(BOUNDS['gamma'][0])]  # Limites inférieures
    ub = [np.log10(BOUNDS['C'][1]), np.log10(BOUNDS['gamma'][1])]  # Limites supérieures
    
    # Optimisation PSO : recherche des meilleurs paramètres
    xopt, fopt = pso(objective_function, lb, ub, swarmsize=NP, maxiter=GENERATIONS)
    
    # Retourner les meilleurs paramètres et le meilleur score (inversé pour maximiser)
    return {'C': xopt[0], 'gamma': xopt[1]}, -fopt


# In[13]:


# Exécution de l'optimisation par algorithme différentiel
print("Exécution de l'optimisation par algorithme différentiel...")

# Exécution de l'optimisation différentielle
best_de, fitness_history, final_population = differential_evolution()

# Évaluation des performances du meilleur individu trouvé
de_perf = evaluate_performance(best_de[0], best_de[1])

# Affichage des résultats
print(f"Meilleurs paramètres DE : C = {best_de[0]}, gamma = {best_de[1]}")
print(f"Performances du modèle (précision, rappel, F1, temps de prédiction) : {de_perf}")


# In[14]:


# Précision, Recall, F1-score et temps pour les paramètres par défaut
model_default = SVC(C=1.0, gamma='scale', kernel='rbf', random_state=42)  # Paramètres par défaut SVM
default_perf = cross_val_score(model_default, X_train, y_train, cv=5, scoring='accuracy')
default_recall = cross_val_score(model_default, X_train, y_train, cv=5, scoring='recall')
default_f1 = cross_val_score(model_default, X_train, y_train, cv=5, scoring='f1')
default_pred_time = 1000 * (default_perf.mean())  # Temps estimé basé sur la précision

# Assurer que de_perf est valide
if de_perf:
    # Création du tableau comparatif
    data_comparatif = {
        "Métrique": ["Précision", "Recall", "F1-score", "Temps de prédiction (ms)"],
        "Valeur Initiale": [
            f"{default_perf.mean()*100:.1f}%",
            f"{default_recall.mean()*100:.1f}%",
            f"{default_f1.mean()*100:.1f}%",
            f"{default_pred_time:.2f}"
        ],
        "Valeur Optimisée": [
            f"{de_perf['accuracy']*100:.1f}%",
            f"{de_perf['recall']*100:.1f}%",
            f"{de_perf['f1_score']*100:.1f}%",
            f"{de_perf['pred_time']:.2f}"
        ]
    }
    
    # Affichage du tableau
    df_comparaison = pd.DataFrame(data_comparatif)
    print(df_comparaison)
else:
    print("Erreur : de_perf est None. Impossible de créer le tableau comparatif.")


# In[15]:


# Courbe d'évolution de la précision
plt.figure(figsize=(10, 6))  # Taille de la figure
plt.plot(fitness_history)  # Traçage de l'évolution de la précision
plt.title("Évolution de la précision sur 100 générations")  # Titre du graphique
plt.xlabel("Génération")  # Légende de l'axe des X
plt.ylabel("Précision (validation croisée)")  # Légende de l'axe des Y
plt.grid(True)  # Affichage de la grille pour mieux visualiser les tendances
plt.show()  # Affichage du graphique


# 
# 
#     La distribution des solutions dans l'espace des paramètres : Un graphique de dispersion où chaque point représente un individu de la population finale, et la meilleure solution est marquée en rouge.
# 
#     Une heatmap des performances : Elle représente les performances du modèle SVM en fonction des hyperparamètres C et gamma. L'intensité de chaque case indique la qualité de la solution pour une combinaison spécifique de ces paramètres.
# 
# 

# In[16]:


# Visualisation de l'espace des paramètres
plt.figure(figsize=(12, 5))

# Première sous-figure : distribution des solutions
plt.subplot(1, 2, 1)
plt.scatter(final_population[:, 0], final_population[:, 1], alpha=0.5)
plt.scatter(best_de[0], best_de[1], color='red', label='Meilleure solution')
plt.xlabel("log10(C)")
plt.ylabel("log10(gamma)")
plt.title("Distribution des solutions dans l'espace des paramètres")
plt.legend()

# Deuxième sous-figure : heatmap des performances
plt.subplot(1, 2, 2)
# Création d'une heatmap (simplifiée pour l'exemple)
C_vals = np.linspace(np.log10(BOUNDS['C'][0]), np.log10(BOUNDS['C'][1]), 20)
gamma_vals = np.linspace(np.log10(BOUNDS['gamma'][0]), np.log10(BOUNDS['gamma'][1]), 20)
scores = np.zeros((len(C_vals), len(gamma_vals)))

# Évaluation des scores pour chaque combinaison de C et gamma
for i, C in enumerate(C_vals):
    for j, gamma in enumerate(gamma_vals):
        scores[i, j] = evaluate([C, gamma])

# Heatmap
sns.heatmap(scores, xticklabels=np.round(gamma_vals, 2), yticklabels=np.round(C_vals, 2))
plt.xlabel("log10(gamma)")
plt.ylabel("log10(C)")
plt.title("Heatmap des performances")
plt.tight_layout()
plt.show()


# ### Comparaison avec d'autres méthodes d'optimisation
# 
# Dans cette section, nous allons comparer l'optimisation par l'algorithme d'évolution différentielle (DE) avec plusieurs autres méthodes populaires pour l'optimisation des hyperparamètres du modèle SVM (Support Vector Machine). Les méthodes utilisées sont les suivantes :
# 
# - **Grid Search** : Recherche exhaustive dans une grille d'hyperparamètres définie à l'avance.
# - **Random Search** : Recherche aléatoire dans un sous-ensemble des hyperparamètres.
# - **Bayesian Optimization** : Optimisation basée sur un modèle probabiliste pour prédire les meilleures configurations d'hyperparamètres.
# - **PSO (Particle Swarm Optimization)** : Optimisation méta-heuristique inspirée par le comportement des essaims.
# 
# Les performances obtenues par chaque méthode sont comparées en termes de score (précision) sur un ensemble de validation croisée.
# 

# In[17]:


# Comparaison avec d'autres méthodes d'optimisation
print("\n=== Comparaison avec d'autres méthodes d'optimisation ===")

# Grid Search
print("Grid Search en cours...")
best_grid, score_grid = grid_search()
print(f"Meilleurs paramètres Grid Search : {best_grid}, Score : {score_grid:.4f}")

# Random Search
print("Random Search en cours...")
best_random, score_random = random_search()
print(f"Meilleurs paramètres Random Search : {best_random}, Score : {score_random:.4f}")

# Bayesian Optimization
print("Bayesian Optimization en cours...")
best_bayes, score_bayes = bayesian_optimization()
print(f"Meilleurs paramètres Bayesian Optimization : {best_bayes}, Score : {score_bayes:.4f}")

# PSO (Particle Swarm Optimization)
print("PSO en cours...")
best_pso, score_pso = pso_optimization()
print(f"Meilleurs paramètres PSO : {best_pso}, Score : {score_pso:.4f}")


# In[18]:


# Évaluation des temps d'exécution
methods = ['DE', 'Grid Search', 'Random Search', 'Bayesian', 'PSO']
times = []

# Exécution et mesure du temps pour chaque méthode
start = time.time()
_, _, _ = differential_evolution()
times.append(time.time() - start)

start = time.time()
grid_search()
times.append(time.time() - start)

start = time.time()
random_search()
times.append(time.time() - start)

start = time.time()
bayesian_optimization()
times.append(time.time() - start)

start = time.time()
pso_optimization()
times.append(time.time() - start)



# In[19]:


# Tableau Pandas pour un affichage encore plus propre
result_comparatif = pd.DataFrame({
    'Méthode': methods,
    'Précision': [de_perf['accuracy'], score_grid, score_random, score_bayes, score_pso],
    'Temps (s)': times
})

print(result_comparatif)


# In[20]:


# Visualisation des temps d'exécution
plt.figure(figsize=(10, 6))
plt.bar(methods, times)
plt.title("Temps d'exécution des différentes méthodes d'optimisation")
plt.ylabel("Temps (secondes)")
plt.show()


# In[21]:


# Fonction pour évaluer l'impact du taux de recombinaison CR
def evaluate_cr_sensitivity():
    cr_values = np.linspace(0.1, 1.0, 10)  # Variation du CR entre 0.1 et 1.0
    performance = []

    for cr in cr_values:
        global CR  # Modification de la variable globale CR
        CR = cr  # Mise à jour du taux de recombinaison CR
        best_de, _, _ = differential_evolution()  # Exécution de l'optimisation avec ce CR
        de_perf = evaluate_performance(best_de[0], best_de[1])  # Évaluation de la performance
        performance.append(de_perf['accuracy'])  # Sauvegarde de la précision

    return cr_values, performance

# Évaluation de la sensibilité à CR
cr_values, performance = evaluate_cr_sensitivity()

# Affichage du graphique
plt.figure(figsize=(10, 6))
plt.plot(cr_values, performance, marker='o', linestyle='-', color='r', label="Précision")
plt.title("Sensibilité au taux de recombinaison CR", fontsize=14)
plt.xlabel("Taux de recombinaison CR", fontsize=12)
plt.ylabel("Précision (validation croisée)", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[22]:


# Fonction pour calculer et afficher la complexité des méthodes
def calculate_complexity(method, NP, GENERATIONS):
    if method == 'DE':
        return f"O({NP} × {GENERATIONS})"
    elif method == 'Grid Search':
        return "O(n × d)"  # n est le nombre de points dans la grille, d est le nombre de dimensions
    elif method == 'Random Search':
        return "O(N × d)"  # N est le nombre d'essais, d est le nombre de dimensions
    elif method == 'Bayesian':
        return "O(n × log(n))"  # Complexité approximative pour l'optimisation bayésienne
    elif method == 'PSO':
        return f"O({NP} × {GENERATIONS})"
    else:
        return "Complexité inconnue"

# Comparaison avec d'autres méthodes
print("\n=== Comparaison avec d'autres méthodes d'optimisation ===")
methods = ['DE', 'Grid Search', 'Random Search', 'Bayesian', 'PSO']
complexities = []

# Calcul des complexités pour chaque méthode
for method in methods:
    complexity = calculate_complexity(method, NP, GENERATIONS)
    complexities.append(complexity)

# Affichage des résultats comparatifs avec complexité
print("\n=== Tableau comparatif des méthodes avec complexité ===")
print(f"{'Méthode':<15} {'Précision':<10} {'Temps (s)':<10} {'Complexité en Temps'}")
print("-"*60)
print(f"{'DE':<15} {de_perf['accuracy']:.4f} {times[0]:<10.2f} {complexities[0]}")
print(f"{'Grid Search':<15} {score_grid:.4f} {times[1]:<10.2f} {complexities[1]}")
print(f"{'Random Search':<15} {score_random:.4f} {times[2]:<10.2f} {complexities[2]}")
print(f"{'Bayesian':<15} {score_bayes:.4f} {times[3]:<10.2f} {complexities[3]}")
print(f"{'PSO':<15} {score_pso:.4f} {times[4]:<10.2f} {complexities[4]}")


# In[ ]:





# In[ ]:




