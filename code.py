import numpy as np

# ==========================================
# 1. IMPURITY & LOSS METRICS
# ==========================================

def calculate_gini(y):
    """Calculates Gini Impurity for classification."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

def calculate_mse(y):
    """Calculates Mean Squared Error (variance) for regression."""
    if len(y) == 0:
        return 0.0
    return float(np.var(y))

def compute_impurity(y, task):
    """Routes target evaluation to Gini (classification) or MSE (regression)."""
    return calculate_gini(y) if task == 'classification' else calculate_mse(y)

# ==========================================
# 2. TREE BUILDING & SPLITTING LOGIC
# ==========================================

def create_leaf_node(y, task):
    """Returns a dictionary representing a leaf node state."""
    if len(y) == 0:
        val = 0.0
    elif task == 'classification':
        vals, counts = np.unique(y, return_counts=True)
        val = vals[np.argmax(counts)]
    else:
        val = float(np.mean(y))
    return {
        'is_leaf': True,
        'value': val
    }

def find_best_split(X, y, feature_indices, min_samples_leaf, task):
    """Iterates features and thresholds to maximize impurity gain."""
    n_samples = len(y)
    current_impurity = compute_impurity(y, task)
    
    best_gain = -1.0
    best_feature = None
    best_threshold = None
    best_left_mask = None
    best_right_mask = None

    for feat_idx in feature_indices:
        column = X[:, feat_idx]
        thresholds = np.unique(column)
        
        for thresh in thresholds:
            left_mask = column <= thresh
            right_mask = ~left_mask
            
            n_left, n_right = np.sum(left_mask), np.sum(right_mask)
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
                
            imp_left = compute_impurity(y[left_mask], task)
            imp_right = compute_impurity(y[right_mask], task)
            
            weighted_imp = (n_left / n_samples) * imp_left + (n_right / n_samples) * imp_right
            gain = current_impurity - weighted_imp
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feat_idx
                best_threshold = thresh
                best_left_mask = left_mask
                best_right_mask = right_mask

    return best_gain, best_feature, best_threshold, best_left_mask, best_right_mask

def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, task='classification'):
    """Recursively constructs decision tree represented as nested dictionaries."""
    n_samples, n_features = X.shape
    
    is_pure = (task == 'classification' and len(np.unique(y)) <= 1)
    reached_max_depth = (max_depth is not None and depth >= max_depth)
    too_few_samples = (n_samples < min_samples_split)
    
    if is_pure or reached_max_depth or too_few_samples:
        return create_leaf_node(y, task)

    # Subsample features for Random Forest split-level randomness
    feature_indices = np.arange(n_features)
    if max_features is not None and max_features < n_features:
        feature_indices = np.random.choice(n_features, size=max_features, replace=False)

    best_gain, best_feat, best_thresh, left_mask, right_mask = find_best_split(
        X, y, feature_indices, min_samples_leaf, task
    )

    if best_gain <= 1e-7 or best_feat is None:
        return create_leaf_node(y, task)

    left_subtree = build_tree(
        X[left_mask], y[left_mask], depth + 1, max_depth, 
        min_samples_split, min_samples_leaf, max_features, task
    )
    right_subtree = build_tree(
        X[right_mask], y[right_mask], depth + 1, max_depth, 
        min_samples_split, min_samples_leaf, max_features, task
    )

    return {
        'is_leaf': False,
        'feature': best_feat,
        'threshold': best_thresh,
        'n_samples': n_samples,
        'impurity_reduction': best_gain * n_samples,
        'left': left_subtree,
        'right': right_subtree
    }

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================

def predict_single_sample(tree, x):
    """Traverses nested dictionary tree for a single input instance."""
    if tree['is_leaf']:
        return tree['value']
    if x[tree['feature']] <= tree['threshold']:
        return predict_single_sample(tree['left'], x)
    else:
        return predict_single_sample(tree['right'], x)

def predict_tree(tree, X):
    """Generates predictions for a matrix of samples."""
    return np.array([predict_single_sample(tree, x) for x in X])

# ==========================================
# 4. RANDOM FOREST IMPLEMENTATION
# ==========================================

def train_random_forest(X, y, n_trees=20, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, task='classification'):
    """Trains an ensemble of procedural trees using bootstrap sampling."""
    forest = []
    n_samples, n_total_features = X.shape

    if max_features is None:
        if task == 'classification':
            max_features = max(1, int(np.sqrt(n_total_features)))
        else:
            max_features = max(1, int(n_total_features / 3))

    for _ in range(n_trees):
        # Bootstrap sampling (sampling with replacement)
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[boot_idx], y[boot_idx]
        
        tree = build_tree(
            X_boot, y_boot, depth=0, max_depth=max_depth, 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            max_features=max_features, task=task
        )
        forest.append(tree)

    return forest

def predict_random_forest(forest, X, task='classification'):
    """Aggregates predictions from all trees via majority voting or averaging."""
    tree_preds = np.array([predict_tree(tree, X) for tree in forest])
    
    if task == 'classification':
        final_preds = []
        for i in range(X.shape[0]):
            vals, counts = np.unique(tree_preds[:, i], return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
        return np.array(final_preds)
    else:
        return np.mean(tree_preds, axis=0)
