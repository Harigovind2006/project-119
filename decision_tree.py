import math
def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    
    counts = {}  
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def max_label(dict):
    max_count = 0
    label = ""

    for key, value in dict.items():
        if dict[key] > max_count:
            max_count = dict[key]
            label = key

    return label

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows
def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def entropy(rows):
    entries = class_counts(rows)
    avg_entropy = 0
    size = float(len(rows))
    for label in entries:
        prob = entries[label] / size
        avg_entropy = avg_entropy + (prob * math.log(prob, 2))
    return -1*avg_entropy
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

def find_best_split(rows, header):
    best_gain = 0  
    best_question = None  
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1 
    for col in range(n_features): 

        values = set([row[col] for row in rows])  

        for val in values: 
            question = Question(col, val, header)

            
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    def __init__(self, rows, id, depth):
        self.predictions = class_counts(rows)
        self.predicted_label = max_label(self.predictions)
        self.id = id
        self.depth = depth


class Decision_Node:
    
    def __init__(self,
                 question,
                 true_branch,
                 false_branch,
                 depth,
                 id,
                 rows):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows
def build_tree(rows, header, depth=0, id=0):
    gain, question = find_best_split(rows, header)
    if gain == 0:
        return Leaf(rows, id, depth)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows, header, depth + 1, 2 * id + 2)
    false_branch = build_tree(false_rows, header, depth + 1, 2 * id + 1)
    return Decision_Node(question, true_branch, false_branch, depth, id, rows)

def prune_tree(node, prunedList):
        
    if isinstance(node, Leaf):
        return node
    if int(node.id) in prunedList:
        return Leaf(node.rows, node.id, node.depth)

    
    node.true_branch = prune_tree(node.true_branch, prunedList)

    
    node.false_branch = prune_tree(node.false_branch, prunedList)
    return node
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predicted_label

   
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Leaf id: " + str(node.id) + " Predictions: " + str(node.predictions) + " Label Class: " + str(node.predicted_label))
        return
    print(spacing + str(node.question) + " id: " + str(node.id) + " depth: " + str(node.depth))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def getLeafNodes(node, leafNodes =[]):
    if isinstance(node, Leaf):
        leafNodes.append(node)
        return
    getLeafNodes(node.true_branch, leafNodes)
    getLeafNodes(node.false_branch, leafNodes)

    return leafNodes
def getInnerNodes(node, innerNodes =[]):
    if isinstance(node, Leaf):
        return

    innerNodes.append(node)

    
    getInnerNodes(node.true_branch, innerNodes)
    getInnerNodes(node.false_branch, innerNodes)
    return innerNodes

def computeAccuracy(rows, node):

    count = len(rows)
    if count == 0:
        return 0

    accuracy = 0
    for row in rows:
        
        if row[-1] == classify(row, node):
            accuracy += 1
    return round(accuracy/count, 2)
