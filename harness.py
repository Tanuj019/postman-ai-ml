import numpy as np
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from tree_rf import build_tree, predict_tree, train_random_forest, predict_random_forest

def run_tests():
    print("==================================================")
    print("     TASK 4: CORRECTNESS AND BENCHMARK HARNESS    ")
    print("==================================================\n")

    # ----------------------------------------------------
    # TEST 1: Match against scikit-learn DecisionTreeClassifier
    # ----------------------------------------------------
    iris = load_iris()
    X, y = iris.data, iris.target
    
    custom_tree = build_tree(X, y, max_depth=3, task='classification')
    custom_preds = predict_tree(custom_tree, X)
    
    sk_tree = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
    sk_tree.fit(X, y)
    sk_preds = sk_tree.predict(X)

    match_pct = np.mean(custom_preds == sk_preds) * 100
    print(f"[TEST 4.6] Sklearn Equivalence Check:")
    print(f"  - Custom Tree Accuracy : {np.mean(custom_preds == y)*100:.2f}%")
    print(f"  - Sklearn Tree Accuracy: {np.mean(sk_preds == y)*100:.2f}%")
    print(f"  - Prediction Match Rate: {match_pct:.2f}%")
    assert match_pct == 100.0, "FAILED: Custom tree predictions do not match sklearn!"
    print("  -> STATUS: PASS\n")

    # ----------------------------------------------------
    # TEST 2: Overfitting & Regularization Demonstration
    # ----------------------------------------------------
    X_noisy, y_noisy = make_classification(
        n_samples=150, n_features=15, n_informative=5, 
        n_clusters_per_class=2, flip_y=0.2, random_state=42
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X_noisy, y_noisy, test_size=0.3, random_state=42)

    tree_unrestricted = build_tree(X_tr, y_tr, max_depth=None, min_samples_split=2, task='classification')
    tr_acc_unres = np.mean(predict_tree(tree_unrestricted, X_tr) == y_tr)
    te_acc_unres = np.mean(predict_tree(tree_unrestricted, X_te) == y_te)

    tree_reg = build_tree(X_tr, y_tr, max_depth=4, min_samples_split=10, min_samples_leaf=4, task='classification')
    tr_acc_reg = np.mean(predict_tree(tree_reg, X_tr) == y_tr)
    te_acc_reg = np.mean(predict_tree(tree_reg, X_te) == y_te)

    print(f"[TEST 4.3 & 4.4] Overfitting & Regularization:")
    print(f"  - Unrestricted Tree -> Train Acc: {tr_acc_unres*100:.1f}%, Test Acc: {te_acc_unres*100:.1f}%")
    print(f"  - Regularized Tree   -> Train Acc: {tr_acc_reg*100:.1f}%, Test Acc: {te_acc_reg*100:.1f}%")
    assert te_acc_reg >= te_acc_unres, "Regularization failed to improve test accuracy!"
    print("  -> STATUS: PASS\n")

    # ----------------------------------------------------
    # TEST 3: Decision Tree vs Random Forest Variance
    # ----------------------------------------------------
    tree_test_accs, rf_test_accs = [], []
    for seed in range(10):
        np.random.seed(seed)
        X_t, X_v, y_t, y_v = train_test_split(X_noisy, y_noisy, test_size=0.4, random_state=seed)
        
        single_t = build_tree(X_t, y_t, max_depth=5, task='classification')
        rf_model = train_random_forest(X_t, y_t, n_trees=25, max_depth=5, task='classification')
        
        tree_test_accs.append(np.mean(predict_tree(single_t, X_v) == y_v))
        rf_test_accs.append(np.mean(predict_random_forest(rf_model, X_v) == y_v))

    print(f"[TEST 4.7] Single Tree vs Random Forest Variance across 10 Runs:")
    print(f"  - Single Tree Test Acc Mean: {np.mean(tree_test_accs)*100:.2f}% | Std Dev: {np.std(tree_test_accs)*100:.2f}%")
    print(f"  - Random Forest Test Acc Mean: {np.mean(rf_test_accs)*100:.2f}% | Std Dev: {np.std(rf_test_accs)*100:.2f}%")
    assert np.std(rf_test_accs) <= np.std(tree_test_accs), "Random Forest variance higher than single tree!"
    print("  -> STATUS: PASS\n")

    # ----------------------------------------------------
    # TEST 4: Regression Functionality Check
    # ----------------------------------------------------
    diab = load_diabetes()
    X_d, y_d = diab.data[:150], diab.target[:150]
    reg_tree = build_tree(X_d, y_d, max_depth=3, task='regression')
    reg_preds = predict_tree(reg_tree, X_d)
    mse_val = np.mean((reg_preds - y_d) ** 2)
    
    print(f"[TEST 4.2] Regression Support:")
    print(f"  - Target Variance: {np.var(y_d):.2f}")
    print(f"  - Tree Model MSE:  {mse_val:.2f}")
    assert mse_val < np.var(y_d), "Regression tree failed to fit data!"
    print("  -> STATUS: PASS\n")

    print("==================================================")
    print("          ALL TEST SUITES PASSED SUCCESSFULLY     ")
    print("==================================================")

if __name__ == '__main__':
    run_tests()
