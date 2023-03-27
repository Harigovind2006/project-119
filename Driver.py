from DecisionTree import *
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('data.csv')
header = list(df.columns)

lst = df.values.tolist()
trainDF, testDF = model_selection.train_test_split(lst, test_size=0.2)


t = build_tree(trainDF, header)

print("\nLeaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

print("\nNon-leaf nodes ****************")
innerNodes = getInnerNodes(t)

for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))


maxAccuracy = computeAccuracy(testDF, t)
print("\nTree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")
print_tree(t)


nodeIdToPrune = -1
for node in innerNodes:
    if node.id != 0:
        prune_tree(t, [node.id])
        currentAccuracy = computeAccuracy(testDF, t)
        print("Pruned node_id: " + str(node.id) + " to achieve accuracy: " + str(currentAccuracy*100) + "%")
        if currentAccuracy > maxAccuracy:
            maxAccuracy = currentAccuracy
            nodeIdToPrune = node.id
        t = build_tree(trainDF, header)
        if maxAccuracy == 1:
            break
    
if nodeIdToPrune != -1:
    t = build_tree(trainDF, header)
    prune_tree(t, [nodeIdToPrune])
    print("\nFinal node Id to prune (for max accuracy): " + str(nodeIdToPrune))
else:
    t = build_tree(trainDF, header)
    print("\nPruning strategy did'nt increased accuracy")