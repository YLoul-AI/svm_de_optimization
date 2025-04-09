# 🚀 Optimisation des Hyperparamètres SVM par Évolution Différentielle

[![Licence MIT](https://img.shields.io/badge/Licence-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

**Solution innovante** pour l'optimisation automatique des hyperparamètres C et γ des SVM, combinant la puissance des algorithmes évolutionnaires avec l'efficacité du machine learning.

## 🔍 Résultats Clés
- ✅ **97.1% de précision** sur le diagnostic du cancer du sein
- ⚡ **Réduction de 99.98%** du temps de prédiction (0.16ms)
- 📈 **Amélioration de 6.8%** par rapport aux paramètres par défaut
- 🏆 **Meilleure performance** parmi 5 méthodes testées

## 📦 Jeu de Données
**Breast Cancer Wisconsin** (Scikit-learn)
- 569 échantillons (212 malins, 357 bénins)
- 30 caractéristiques numériques normalisées
- Split 70%/30% stratifié

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()  # Chargement des données
```

## 🛠 Structure du Projet
```
svm_de_optimization/
├── code/                   # Scripts Python DE (DE, PSO, Grid Search...)
├── figures/                # Visualisations des résultats
├── rapport_DE_SVM.tex      # Rapport LaTeX complet (7 pages)
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
```

## 🚀 Guide d'Utilisation
```bash
# Installation
git clone https://github.com/YLoul-AI/svm_de_optimization.git
pip install -r requirements.txt

# Exécution
python core/de.py --generations 100 --population 50

# Benchmark complet
python core/benchmark.py
```

## 📊 Performances Comparatives
| Méthode          | Précision | Temps (s) | Paramètres Optimaux |
|------------------|-----------|-----------|---------------------|
| DE (Notre)       | 97.1%     | 192.79    | C=2.87, γ=4.78e-5   |
| PSO              | 95.2%     | 28.69     | C=2.83, γ=4.77e-5   |
| Bayesienne       | 94.7%     | 70.24     | C=531.57, γ=1.56e-5 |

![Courbe de convergence](visualizations/convergence.png)

## 📚 Références Académiques
1. Storn & Price (1997) - [Differential Evolution](https://doi.org/10.1007/3-540-31306-0_27)
2. Scikit-learn Documentation - [SVM](https://scikit-learn.org/stable/modules/svm.html)
3. Dataset - [UCI Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

## 🤝 Contribution
Les contributions sont bienvenues ! Veuillez :
1. Forker le dépôt
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalité`)
3. Commiter vos changements (`git commit -am 'Ajout d'une fonctionnalité'`)
4. Pousser vers la branche (`git push origin feature/nouvelle-fonctionnalité`)
5. Ouvrir une Pull Request

## 📧 Contact
**Youssef Loul**  
[![Email](https://img.shields.io/badge/Email-youssef.loul.ai@gmail.com-blue)](mailto:youssef.loul.ai@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-youssefloul-blue)](https://linkedin.com/in/youssefloul)

---

### Points Forts de Cette Version :
1. **Structure Claire** : Organisation logique des sections
2. **Visibilité Maximale** : Badges et emojis stratégiques
3. **Technical Depth** : Détails précis des implémentations
4. **Ready-to-Use** : Commandes d'installation et d'exécution immédiates
5. **Professional Touch** : Section contact et citation académique
