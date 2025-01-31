import csv
from math import sqrt
import sys

def DataFrame(val):
    print("",*range(0, len(val[0])), sep="\t")
    


if (len(sys.argv) > 1):
    try:
        with open(sys.argv[1], newline='') as csvfile:

            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            name=[]
            line=0
            size = 0
            val= []
            for row in spamreader:
                if line ==0:
                    size = len(row)
                    name = row
                    for i in range(len(name)):
                        if (len(name[i]) > 12):
                            name[i] = name[i][:10] + "..."
                        
                    line +=1
                    continue
                element = [None]*size
                for i,elem in enumerate(row[6:]):
                    if str(elem) == "":
                        continue
                    elem = float(elem)
                    element[i] = elem
                
                val.append(element)
                tab = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
            std=[]
            for elem in val:
                for j in range(0,size):
                    if (elem[j]):
                        tab[j].append(elem[j])
            print("\t\t",end="")
            for i in range(6,19):
                print(f"{name[i]:>9}",end="\t")
            print("\ncount", end="\t\t")
            for t in tab:
                t.sort()
                print(f"{len(t):>7}", end="\t\t")  
            print("\nmean", end="\t\t")
            for t in tab:
                mean=0
                for val in t:
                    mean+=val
                mean = round(mean/len(tab[0]),6)
                std.append(round(sqrt(abs(mean)),6))
                print(f"{mean:>10}", end="\t")
            print("\nstd", end="\t\t")
            for i in range(0,len(tab)):
                print(f"{std[i]:>10}",end="\t")
            print("\nmin", end="\t\t")
            for t in tab:
                print(f"{round(t[0],6):>10}",end="\t")
            print("\n25%", end="\t\t")
            for t in tab:
                print(f"{round(t[round(len(t)/4)],6):>10}",end="\t")
            print("\n50%", end="\t\t")
            for t in tab:
                print(f"{round(t[round(len(t)/2)],6):>10}",end="\t")
            print("\n75%", end="\t\t")
            for t in tab:
                print(f"{round(t[round(len(t)/3)*2],6):>10}",end="\t")
            print("\nmax", end="\t\t")
            for t in tab:
                print(f"{round(t[-1],6):>10}",end="\t")
            
    except:
        print("Fatal error while opening csv")
else:
    print("Fatal error")