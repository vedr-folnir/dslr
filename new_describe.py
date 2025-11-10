import csv
from math import sqrt
import sys


def is_number(s):
    if s.startswith("-"):
        s = s[1:]  # Remove leading minus
    return s.replace(".", "", 1).isdigit() and s.count(".") <= 1


def p_value(what, tab, where):
    print("\n", what, end="\t", sep="")
    for unit in tab:
        try:
            print(f"{round(float(unit[round(len(unit)*where)]),6):>12}", end="\t")
        except:
            print(f"{round(float(unit[-1]),6):>12}", end="\t")


if (len(sys.argv) != 2):
    print("error wrong arg")
    exit()
    
with open(sys.argv[1], newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    names = []
    size = 0
    tab = []
    usable = []
    std = []
    for i, row in enumerate(spamreader):
        if i == 0:
            size = len(row) - 1
            tab = [[] for _ in range(size)]
            names = row[1:]
            for i in range(len(names)):
                if (len(names[i]) > 12):
                    names[i] = names[i][:9] + "..."
            continue
        element = [None]*size
        for i, elem in enumerate(row[1:]):
            if str(elem) == "":
                continue
            elem = elem   
            tab[i].append(elem)
    print("\t",end="")
    for i in range(size):
        try:
            if (is_number(tab[i][0])):
                usable.append(tab[i])
                print(f"{names[i]:>12}", end="\t")
        except:
            continue
    print("\ncount", end="")
    for t in usable:
        for i in range(len(t)):
            t[i] = float(t[i])
        t.sort()
        print(f"{len(t):>12}", end="\t")  
    print("\nmean", end="\t")
    for t in usable:
        mean = 0
        for val in t:
            mean += float(val)
        mean = round(mean/len(usable[0]),6)
        std.append(round(sqrt(abs(mean)),6))
        print(f"{mean:>12}", end="\t")
    print("\nstd", end="\t")
    for i in range(0, len(usable)):
        print(f"{std[i]:>12}", end="\t")
    p_value("min", usable, 0)
    p_value("25%", usable, 1/4)
    p_value("50%", usable, 2/4)
    p_value("75%", usable, 3/4)
    p_value("max", usable, 1)
