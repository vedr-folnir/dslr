import csv


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons




def open_dataset(path: str):
    """ 
        ouvre le dataset du path
        recup les infos dans un bloc divise par matiere
        remplace les cases vides par None
        ne prend pas en compte les erreurs
    """
    
    csvfile = open(path, newline='')
    file = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    for i, line in enumerate(file):
        if i == 0:
            dataset = [[] for _ in range(len(line))]
            pass
        for y, elem in enumerate(line):
            if len(elem) == 0:
                dataset[y].append(None)
                continue
            dataset[y].append(elem)
    return dataset

def correct_err(dataset, index):
    """
        prend un dataset et remplace les None par la valeur moyenne 
        renvoi le dataset corriger
    """
    for i, data in enumerate(dataset[index:], index):
        moy = np.mean([float(v) for v in data[1:] if v is not None])
        dataset[i] = [v if v is not None else moy for v in dataset[i]]
    return dataset
    

def sort_by_kind(dataset, index):
    """
        prend un dataset et les tri par rapport au donner du champs
        ex: je veux trier par maisons tu donne l'index du champ maison
            le prog cherche le nombre de diff maisons et les trie
            dataset => maison[dataset,Ravenclaw,Slytherin,Gryffindor,Hufflepuff]
        le retour est un tableau de taille variable dans un ordre radom a cause du set()
        avec comme 0 le dataset de base
    """
    # sorted = [[] for _ in range(len(set(dataset[index])))]
    # names = list(set(dataset[index][1:]))
    # print(names)
    # for i in range(len(dataset[index])):
    #     for n, name in enumerate(names):
    #         if name == dataset[index][i]:
    #             stud = []
    #             for elem in range(len(dataset)):
    #                 stud.append(dataset[elem][i])
    #             sorted[n].append(stud)  
    
    names = list(set(dataset[index][1:]))
    sorted = [[] for _ in range(len(names))]
    for stud in range(len(dataset[index])):
        # print(i)
        if dataset[index][stud] in names:
            sorted[names.index(dataset[index][stud])].append(stud)
            # print(stud)
            
        
    return sorted, names

def get_data(dataset, index, to):
    """
        retourne une liste de valeurs du dataset a l'index de chaque membre de to
    """
    values = []
    for elem in to:
        values.append(dataset[index][elem])
    return values

def cor(dataset):
    """
        fait la correlation entre chaque valeurs et renvoi l'index des 2 plus similaires
    """
    max = [0]
    for i in range(len(dataset)):
        fam = [float(val) for val in dataset[i][1:]]
        for j in range(len(dataset[i:])):
            fim = [float(val) for val in dataset[j][1:]]
            corco = abs(np.corrcoef(fim,fam)[0,1])
            if corco > max[0] and i != j:
                max = [corco, i, j]       
    
    val1 = [float(val) for val in dataset[max[1]][1:]]
    val2 = [float(val) for val in dataset[max[2]][1:]]
    
    plt.scatter(val1, val2)
    plt.title("Valeurs en colonne")
    plt.xlabel(study[max[1]])
    plt.ylabel(study[max[2]])
    plt.grid(True, axis='y')
    plt.show()
 
    return max




data = open_dataset("datasets/dataset_train.csv")
# print(data[6][21])

data = correct_err(data, 6)
# print(data[6])

sorted, names = sort_by_kind(data, 1)
# print(sorted, names)

study = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]

fim = get_data(data, 6 + study.index("Astronomy"), sorted[names.index("Slytherin")])
fim = [float(val) for val in fim]

coef = cor(data[6:])
print(coef)
# print(fim)


























# n_colonnes = 13
# couleurs = ['red', 'blue', 'green', 'orange']

# # Créer la figure et les axes
# fig, ax = plt.subplots()
# fig.subplots_adjust(left=0.2)

# # Afficher les points pour chaque groupe
# x = np.arange(1, 14)
# scatters = {}

# # for maison in range(len(names)):
# #     data = sorted[maison]
# #     print(data)
# #     # on trace chaque étudiant comme une ligne de points
# #     plots = []
# #     for student_data in data[:10]:
# #         s = ax.plot(x, student_data[6:], 'o', color=couleurs[maison], alpha=0.6)[0]
# #         plots.append(s)
# #     scatters[maison] = plots

# ax.set_title("Données des étudiants (par maison)")
# ax.set_xlabel("Colonne (1 à 19)")
# ax.set_ylabel("Valeur")
# ax.set_xticks(x)
# ax.set_xlim(1, 19)
# ax.grid(True)

# rax = plt.axes([0.02, 0.4, 0.15, 0.2])  # position des boutons
# labels = names
# visibility = [True] * len(names)
# check = CheckButtons(rax, labels, visibility)

# # Fonction pour mettre à jour la visibilité
# def func(label):
#     index = names.index(label)
#     visible = not scatters[label][0].get_visible()
#     for s in scatters[label]:
#         s.set_visible(visible)
#     plt.draw()

# check.on_clicked(func)

# plt.show()
