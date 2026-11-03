import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Chỉ dành cho leaf node (Giá trị dự đoán)

class DecisionTreeRegressor:
    """
    Cây quyết định hồi quy (Decision Tree Regressor) tự lập trình từ đầu.
    Sử dụng tiêu chí giảm phương sai (Variance Reduction) để rẽ nhánh.
    """
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_vec = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else np.array(y).reshape(-1, 1)
        dataset = np.concatenate((X_mat, y_vec), axis=1)
        self.root = self._build_tree(dataset, curr_depth=0)
        return self

    def _build_tree(self, dataset, curr_depth):
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        
        # Kiểm tra điều kiện dừng
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self._get_best_split(dataset, num_samples, num_features)
            if best_split and best_split["var_red"] > 0:
                left_subtree = self._build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self._build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree)
        
        # Tạo Leaf node
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)

    def _get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_var_red = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self._split(dataset, feature_index, threshold)
                
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_var_red = self._variance_reduction(y, left_y, right_y)
                    
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        return best_split

    def _split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def _variance_reduction(self, parent_y, left_y, right_y):
        weight_l = len(left_y) / len(parent_y)
        weight_r = len(right_y) / len(parent_y)
        reduction = np.var(parent_y) - (weight_l * np.var(left_y) + weight_r * np.var(right_y))
        return reduction

    def _calculate_leaf_value(self, y):
        val = np.mean(y)
        return val

    def predict(self, X):
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        predictions = [self._make_prediction(x, self.root) for x in X_mat]
        return np.array(predictions)

    def _make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)

    def save(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.root, f)

    def load(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            self.root = pickle.load(f)
        return self
