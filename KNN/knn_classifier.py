import numpy as np

# -------------------------
# Dataset
# -------------------------

data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]


# -------------------------
# Step 1: Encoding
# -------------------------

# manually encoding the fruit labels to numbers
# Apple -> 0, Banana -> 1, Orange -> 2

label_map = {
    'Apple': 0,
    'Banana': 1,
    'Orange': 2
}

# reverse map so we can decode predictions back to fruit names
reverse_label_map = {v: k for k, v in label_map.items()}

# separate features and labels
X = []
y = []

for row in data:
    X.append([row[0], row[1], row[2]])   # weight, size, color
    y.append(label_map[row[3]])           # encoded label

X = np.array(X, dtype=float)
y = np.array(y)

print("Feature matrix X:")
print(X)
print("\nLabel vector y:", y)


# --------------------------------------------
# Step 2: Euclidean Distance and other metrics
# --------------------------------------------

def euclidean_distance(point1, point2):
    # formula: sqrt of sum of squared differences
    distance = np.sqrt(np.sum((point1 - point2) ** 2))
    return distance

def manhattan_distance(point1, point2):
    # formula: sum of absolute differences
    distance = np.sum(np.abs(point1 - point2))
    return distance
 
def minkowski_distance(point1, point2, p=3):
    # generalized formula: p-th root of sum of absolute differences raised to power p
    # when p=1 it becomes manhattan, when p=2 it becomes euclidean
    distance = np.sum(np.abs(point1 - point2) ** p) ** (1/p)
    return distance

# -------------------------
# Step 3: KNN Classifier
# -------------------------

class KNN:
    def __init__(self, k=3,metric='euclidean'):
        self.k = k
        self.metric=metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # store the training data
        self.X_train = X
        self.y_train = y

    def get_distance(self, point1, point2):
        # pick the distance function based on self.metric
        if self.metric == 'euclidean':
            return euclidean_distance(point1, point2)
        elif self.metric == 'manhattan':
            return manhattan_distance(point1, point2)
        elif self.metric == 'minkowski':
            return minkowski_distance(point1, point2)
        else:
            print("unknown metric, using euclidean by default")
            return euclidean_distance(point1, point2)

    def predict_one(self, x):
        # calculate distance from x to every training point
        distances = []
        for i in range(len(self.X_train)):
            d = self.get_distance(x, self.X_train[i])
            distances.append((d, self.y_train[i]))
            

        # sort by distance (smallest first)
        distances.sort()

        # pick the k nearest neighbors
        k_nearest = distances[:self.k]

        # count votes for each class
        votes = {}
        for _, label in k_nearest:
            if label not in votes:
                votes[label] = 0
            votes[label] += 1

        # return the class with most votes
        predicted_label = max(votes, key=votes.get)
        return predicted_label

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            pred = self.predict_one(x)
            predictions.append(pred)
        return np.array(predictions)


# -------------------------
# Step 4: Testing
# -------------------------

test_data = np.array([
    [118, 6.2, 0],   # Expected: Banana
    [160, 7.3, 1],   # Expected: Apple
    [185, 7.7, 2]    # Expected: Orange
])

color_map = {0: 'Yellow', 1: 'Red', 2: 'Orange'}

print("\n--- Test Data ---")
for i, sample in enumerate(test_data):
    print(f"Test sample {i+1}: Weight={sample[0]}g, Size={sample[1]}cm, Color={color_map[sample[2]]}")


# create classifier with k=3
knn = KNN(k=3)
knn.fit(X, y)

predictions = knn.predict(test_data)

print("\n--- Predictions (k=3) ---\n---(Euclidean distance)---")
for i, pred in enumerate(predictions):
    fruit_name = reverse_label_map[pred]
    print(f"Test sample {i+1}: {fruit_name}")


# -------------------------
# Step 5: Evaluation
# -------------------------

# expected labels for the test samples
expected_labels = np.array([
    label_map['Banana'],
    label_map['Apple'],
    label_map['Orange']
])

correct = np.sum(predictions == expected_labels)
accuracy = correct / len(expected_labels) * 100
print(f"\nAccuracy on test data: {accuracy}%")

# trying different values of k
print("\n--- Trying different values of k ---")
for k_val in [1, 3, 5]:
    model = KNN(k=k_val)
    model.fit(X, y)
    preds = model.predict(test_data)
    decoded = [reverse_label_map[p] for p in preds]
    print(f"k={k_val} -> Predictions: {decoded}")


# -------------------------
# Bonus: Different Distance Metrics
# -------------------------
 
print("\n--- Trying different distance metrics (k=3) ---")
for metric_name in ['euclidean', 'manhattan', 'minkowski']:
    model = KNN(k=3, metric=metric_name)
    model.fit(X, y)
    preds = model.predict(test_data)
    decoded = [reverse_label_map[p] for p in preds]
    print(f"metric={metric_name} -> Predictions: {decoded}")


# -------------------------
# Bonus: Weighted KNN
# -------------------------
 
# in normal KNN every neighbor gets 1 vote regardless of how close they are
# in weighted KNN closer neighbors get more voting power than farther ones
# the weight of each neighbor is 1/distance
# so if a neighbor is very close (small distance), 1/distance is a big number = more power
# if a neighbor is far (large distance), 1/distance is a small number = less power
 
class WeightedKNN:
    def __init__(self, k=3):  
        self.k = k
        self.X_train = None
        self.y_train = None
 
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
 
    def predict_one(self, x):
        # calculate euclidean distances to all training points
        distances = []
        for i in range(len(self.X_train)):
            d = euclidean_distance(x, self.X_train[i])
            distances.append((d, self.y_train[i]))
 
        # sort by distance
        distances.sort()
 
        # pick k nearest
        k_nearest = distances[:self.k]
 
        # weighted voting — instead of adding 1 vote, we add 1/distance as the vote
        votes = {}
        for distance, label in k_nearest:
 
            # if distance is exactly 0 (perfect match), give it a very large weight
            # we cant do 1/0 as it will cause an error
            if distance == 0:
                weight = 999999
            else:
                weight = 1 / distance   # closer = higher weight
 
            if label not in votes:
                votes[label] = 0
            votes[label] += weight   # add the weight instead of just 1
 
        # the class with the highest total weight wins
        predicted_label = max(votes, key=votes.get)
        return predicted_label
 
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            pred = self.predict_one(x)
            predictions.append(pred)
        return np.array(predictions)
 
 
print("\n--- Weighted KNN using Euclidean Distance (k=3) ---")
wknn = WeightedKNN(k=3)
wknn.fit(X, y)
wpreds = wknn.predict(test_data)
 
for i, pred in enumerate(wpreds):
    fruit_name = reverse_label_map[pred]
    print(f"Test sample {i+1}: {fruit_name}")
 

# compare normal KNN vs weighted KNN side by side
# On this small dataset they'll likely be the same, 
# but on larger noisier datasets weighted KNN often performs better.

print("\n--- Normal KNN vs Weighted KNN (k=3) ---")
normal_preds  = [reverse_label_map[p] for p in knn.predict(test_data)]
weighted_preds = [reverse_label_map[p] for p in wpreds]
 
for i in range(len(test_data)):
    print(f"Test sample {i+1}: Normal={normal_preds[i]}, Weighted={weighted_preds[i]}")


# -------------------------
# User Input Prediction
# -------------------------
 
print("\n--- Predict Your Own Fruit ---")
print("Color codes: 0 = Yellow, 1 = Red, 2 = Orange")
 
weight = float(input("\nEnter weight (in grams): "))
size   = float(input("Enter size (in cm): "))
color  = float(input("Enter color (0, 1, or 2): "))
 
user_input = np.array([[weight, size, color]])
 
pred = reverse_label_map[knn.predict(user_input)[0]]
print(f"\nPredicted fruit: {pred}")
 