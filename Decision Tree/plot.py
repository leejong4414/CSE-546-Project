import numpy as np
import matplotlib as ml
#ml.use('Agg')
import matplotlib.pyplot as plt
from sklearn import tree


train_acc = [] 
validation_acc =[]

train_acc_pca = [] 
validation_acc_pca =[]

with open("decision_tree.txt", "r") as ins:
    i = 0
    for line in ins:
        if i%3 == 1:
            train_acc.append(float(line.rstrip('\n'))*100)
        if i%3 == 2:
            validation_acc.append(float(line.rstrip('\n'))*100)
        i = i + 1
    

with open("decision_tree_pca.txt", "r") as ins:
    i = 0
    for line in ins:
        if i%3 == 1:
            train_acc_pca.append(float(line.rstrip('\n'))*100)
        if i%3 == 2:
            validation_acc_pca.append(float(line.rstrip('\n'))*100)
        i = i + 1
    

plt.title("Decision Tree accuracy")
plt.plot(range(1, 51), train_acc, label = 'train accuracy without PCA', color = 'aqua')
plt.plot(range(1, 51), validation_acc, label = 'validation accuracy without PCA', color = 'aquamarine')

plt.plot(range(1, 51), train_acc_pca, label = 'train accuracy with PCA', color = 'orange')
plt.plot(range(1, 51), validation_acc_pca, label = 'validation accuracy with PCA', color = 'orangered')

plt.scatter(25, 0.899813211071489*100 , label = 'test accuracy witout PCA', color = 'blue')
plt.scatter(13, 0.905586687*100 , label = 'test accuracy with PCA', color = 'brown')
plt.xlabel('depth of decision tree')
plt.xlabel('accuracy')

plt.legend()
plt.savefig('Decision_Tree_accuracy.png')
plt.show()
plt.close()

