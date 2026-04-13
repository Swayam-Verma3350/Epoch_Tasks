import numpy as np

# ---- Step 1: Dataset and Encoding ----

data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
]

# encoding labels to integers manually
label_map = {'Beer': 0, 'Wine': 1, 'Whiskey': 2}
reverse_map = {0: 'Beer', 1: 'Wine', 2: 'Whiskey'}

X = np.array([[row[0], row[1], row[2]] for row in data], dtype=float)
y = np.array([label_map[row[3]] for row in data])

feature_names = ['Alcohol', 'Sugar', 'Color']


# ---- Step 2: Gini Impurity ----

def gini_impurity(labels):
    if len(labels) == 0:
        return 0

    total = len(labels)
    impurity = 1.0

    classes = np.unique(labels)
    for c in classes:
        count = 0
        for i in labels:
           if i == c:
             count += 1
        prob = count / total
        impurity -= prob ** 2

    return impurity

# ---- Bonus: Entropy ----

# formula: -sum(p * log2(p)) for each class
def entropy(labels):
    if len(labels) == 0:
        return 0
 
    total = len(labels)
    result = 0.0
 
    classes = np.unique(labels)
    for c in classes:
        count=0
        for i in labels:
            if i==c:
                count+=1
        prob = count / total
        if prob > 0:
            result -= prob * np.log2(prob)
 
    return result


# ---- Step 3: Best Split Finder ----

def find_best_split(X, y,criterion='gini'):
    best_score = float('inf')
    best_feature = None
    best_threshold = None

    n_samples, n_features = X.shape

    for feature_index in range(n_features):
        sorted_vals = np.unique(X[:, feature_index])

        # for binary features like Color (0 or 1), use 0 and 1 directly
        # for continuous features, use midpoints between consecutive unique values
        if set(sorted_vals) == {0.0, 1.0}:
            thresholds = np.array([0.0, 1.0])
        elif len(sorted_vals) == 1:
            thresholds = sorted_vals
        else:
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2

        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            if criterion == 'gini':
                left_score  = gini_impurity(y_left)
                right_score = gini_impurity(y_right)
            elif criterion== 'entropy':
                left_score  = entropy(y_left)
                right_score = entropy(y_right)
 
            weighted_score = (len(y_left) / n_samples) * left_score + (len(y_right) / n_samples) * right_score
 
            if weighted_score < best_score:
                best_score = weighted_score
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold


# ---- Step 4: Node class and Tree Building ----

class Node:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # only set if leaf node


def build_tree(X, y, depth=0, criterion='gini'):
    
    if len(np.unique(y)) == 1:
        leaf = Node()
        leaf.value = y[0]
        return leaf

    feature, threshold = find_best_split(X, y, criterion)

    if feature is None:
        leaf = Node()
        leaf.value = np.bincount(y).argmax()
        return leaf

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    node = Node()
    node.feature_index = feature
    node.threshold = threshold
    node.left = build_tree(X[left_mask], y[left_mask], depth + 1, )
    node.right = build_tree(X[right_mask], y[right_mask], depth + 1, )

    return node


# ---- Step 5: Prediction ----

def predict_single(node, x):
    if node.value is not None:
        return node.value

    if x[node.feature_index] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)


def predict(tree, X):
    predictions = []
    for x in X:
        pred = predict_single(tree, x)
        predictions.append(pred)
    return np.array(predictions)


# ---- Step 6: Train and Evaluate ----

tree = build_tree(X, y)

# check accuracy on training data
train_preds = predict(tree, X)
accuracy = np.sum(train_preds == y) / len(y)
print(f"Training Accuracy: {accuracy * 100:.1f}%")

# test samples — columns are [Alcohol, Sugar, Color]
test_data = np.array([
    [6.0,  2.1,  0],  # Expected: Beer
    [39.0, 0.05, 1],  # Expected: Whiskey
    [13.0, 1.3,  1]   # Expected: Wine
])

print("\nPredictions on test data:")
test_preds = predict(tree, test_data)
expected_labels = ['Beer', 'Whiskey', 'Wine']
for i in range(len(test_preds)):
    predicted = reverse_map[test_preds[i]]
    expected = expected_labels[i]
    
    if predicted == expected:
        result = "CORRECT"
    else:
        result = "WRONG"
    
    print(f"Sample {i+1}: Predicted = {predicted}, Expected = {expected} => {result}")

# print the tree structure
def print_tree(node, indent=""):
    if node.value is not None:
        print(indent + f"=> Predict: {reverse_map[node.value]}")
        return
    fname = feature_names[node.feature_index]
    print(indent + f"If {fname} <= {node.threshold}:")
    print_tree(node.left,  indent + "  ")
    print(indent + f"Else ({fname} > {node.threshold}):")
    print_tree(node.right, indent + "  ")

print("\nTree Structure:")
print_tree(tree)


# ---- Bonus: Same prediction using Entropy ----
 
tree_entropy = build_tree(X, y, criterion='entropy')
 
train_preds_e = predict(tree_entropy, X)
accuracy_e = np.sum(train_preds_e == y) / len(y)
print("\n---- Bonus: Entropy ----")
print(f"Training Accuracy: {accuracy_e * 100:.1f}%")
 
print("\nPredictions on test data:")
test_preds_e = predict(tree_entropy, test_data)
 
for i in range(len(test_preds_e)):
    predicted = reverse_map[test_preds_e[i]]
    expected  = expected_labels[i]
 
    if predicted == expected:
        result = "CORRECT"
    else:
        result = "WRONG"
 
    print(f"  Sample {i+1}: Predicted = {predicted}, Expected = {expected} => {result}")


# Taking data from user and predicting the drink
print("\n--- Predict Your Own Drink ---")
alcohol = float(input("Enter Alcohol content (%): "))
sugar = float(input("Enter Sugar content (g/L): "))
color = int(input("Enter Color (0 for light, 1 for dark): "))

user_sample = np.array([[alcohol, sugar, color]])
user_pred = predict(tree, user_sample)
print(f"Predicted Drink: {reverse_map[user_pred[0]]}")
