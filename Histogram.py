import csv
from math import sqrt
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def is_number(s):
    if s.startswith("-"):
        s = s[1:]  # enleve le moins
    return s.replace(".", "", 1).isdigit() and s.count(".") <= 1

if (len(sys.argv) != 2):
    print("error wrong number of arguments")

with open(sys.argv[1], newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    names = []
    size = 0
    houses = [[],[],[],[]]
    houses_mean = []
    tab = [[] for _ in range(19)]
    for i, row in enumerate(spamreader):
        if i == 0:
            size = len(row) - 1
            names = row[1:]
            tab = [[] for _ in range(size)]
            for i in range(len(names)):
                if (len(names[i]) > 12):
                    names[i] = names[i][:9] + "..."
            continue
        student = []
        for i,elem in enumerate(row[1:]):
            if str(elem) == "":
                elem = None
            elem=elem
            if (elem != None):
                try:
                    tab[i].append(float(elem))
                except:
                    pass
            student.append(elem)
        if row[1] =="Gryffindor":
            houses[0].append(student)
        elif row[1] == "Hufflepuff":
            houses[1].append(student)
        elif row[1] == "Ravenclaw":
            houses[2].append(student)
        else:
            houses[3].append(student)                
    for house in houses:
        means = [[0,0] for _ in range(size)]
        for stud in house:
            for i,notes in enumerate(stud):
                try:
                    # if (notes==None):
                    means[i][0] += float(notes)
                    means[i][1]+=1
                except:
                    continue
        means = means[5:]
        houses_mean.append(means)
        x=[]
        for i in range(len(means)):
            means[i][0] = means[i][0]/means[i][1]
            x.append(means[i][0])


    min_max=[[float('inf'),float('-inf')] for _ in range(size-5)]
    for i,elem in enumerate(tab[5:]):
        elem.sort()
        min_max[i][0] = elem[0]
        min_max[i][1] = elem[-1]
        
    


    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    colors = sns.color_palette("Set2")

    what_color=['red', 'yellow', 'blue', 'green']
    hat=['/','\\','+','o']
    house_name=["Gryffindor","Hufflepuff","Ravenclaw","Slitherin"]
    for house in range(len(houses_mean)):
        essai = []
        for i,val in enumerate(houses_mean[house]):
            essai.append((val[0] - min_max[i][0]) / (min_max[i][1]-min_max[i][0]) * 100)
        ax.bar(names[5:], essai, width=0.8, edgecolor='b', color=what_color[house], alpha=0.3)
        ax.legend()

    plt.show()
