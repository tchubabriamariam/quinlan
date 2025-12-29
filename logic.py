import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


# 1. Load dataset

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)


# 2. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)


# 3. Train black-box Random Forest

blackbox = RandomForestClassifier(n_estimators=100, random_state=42)
blackbox.fit(X_train, y_train)

# Predictions for fidelity evaluation
y_pred_blackbox = blackbox.predict(X_test)


# 4. Generate surrogate labels

y_train_surrogate = blackbox.predict(X_train)


# 5. Train surrogate interpretable Decision Tree

surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
surrogate.fit(X_train, y_train_surrogate)


# 6. Extract rules from tree (paths → rules)


def extract_tree_rules(tree, feature_names):
    tree_ = tree.tree_
    rules = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            recurse(tree_.children_left[node], path + [(name, "<=", threshold)])
            recurse(tree_.children_right[node], path + [(name, ">", threshold)])
        else:
            label = int(np.argmax(tree_.value[node][0]))
            rules.append((path, label))

    recurse(0, [])
    return rules


rules = extract_tree_rules(surrogate, X_train.columns)


# 7. Convert extracted rules → Prolog-style logic program


def pretty_logic_program(rules):
    program = []

    for path, label in rules:
        if not path:
            program.append(f"class(X, {label}).")
            continue

        body_parts = []
        for feat, op, thr in path:
            pred = feat.lower().replace(" ", "_").replace("(", "").replace(")", "")

            if op == "<=":
                cond = f"{pred}(X, V), V =< {thr:.4f}"
            else:
                cond = f"{pred}(X, V), V > {thr:.4f}"

            body_parts.append(cond)

        rule = "class(X, {label}) ←\n    ".format(label=label)
        rule += ",\n    ".join(body_parts) + "."
        program.append(rule)

    return "\n\n".join(program)


logic_program_str = pretty_logic_program(rules)


print(logic_program_str)


def logic_predict(sample, rules):

    for path, label in rules:
        ok = True
        for feat, op, thr in path:
            val = sample[feat]
            if op == "<=" and not val <= thr:
                ok = False
                break
            if op == ">" and not val > thr:
                ok = False
                break
        if ok:
            return label

    # Default fallback: majority label
    return Counter([label for _, label in rules]).most_common(1)[0][0]


X_test_records = X_test.to_dict(orient="records")
y_pred_logic = [logic_predict(row, rules) for row in X_test_records]

fidelity = np.mean(y_pred_logic == y_pred_blackbox)
accuracy = np.mean(y_pred_logic == y_test)


print(f"Fidelity (surrogate logic program vs black-box): {fidelity:.3f}")
print(f"Accuracy (logic program vs true labels):        {accuracy:.3f}")
print("Number of rules:", len(rules))
