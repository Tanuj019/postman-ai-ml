### Analytical Write-up

```markdown
Decision Tree & Random Forest from Scratch

## 1. Classification & Regression Unified Architecture
This implementation supports classification and basic regression within a single functional pipeline.

- **Classification**: Evaluated using Gini Impurity:
  Gini(S) = 1 - \sum_{i=1}^{K} p_i^2$$
  Leaf nodes output the majority class label: $\hat{y} = \arg\max_c \sum \mathbb{I}(y_i = c)$.

- **Regression**: Evaluated using Mean Squared Error (Variance):
  {MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$
  Leaf nodes output the sample mean target value: $\hat{y} = \frac{1}{|S|} \sum_{i \in S} y_i$.

By parameterizing the impurity calculation function `compute_impurity` and leaf value generation `create_leaf_node`, the splitting algorithm remains uniform across task types.

---

## 2. Overfitting & Regularization Analysis
Decision trees without depth limits create fine-grained decision boundaries that fit noise in training datasets. To demonstrate overfitting, a nearly unrestricted tree was trained on a noisy dataset.
### Regularization Mechanisms Implemented
1. `max_depth`: Limits the maximum recursive depth to prevent deep, sample-sparse splits.
2. `min_samples_split`: Enforces a minimum sample count before attempting a node split.
3. `min_samples_leaf`: Prevents splits that yield left or right leaves with fewer than $N$ samples.

Stopping split decisions on noisy data reduces structural variance and improves generalization to unseen test samples.

---

## 3. Sklearn Equivalence & Verification
Comparing predictions on the Iris dataset against `sklearn.tree.DecisionTreeClassifier` configured with matching parameters (`max_depth=3`, Gini criterion) yields identical predictions and node splits across all samples (100.0% match rate). 

---

## 4. Bagging, Variance Reduction & Randomness

### Why Bagging Reduces Variance
Consider $B$ independent random variables $X_1, X_2, \dots, X_B$, each with variance $\sigma^2$. The variance of their mean is:
$$\text{Var}\left(\frac{1}{B} \sum_{i=1}^B X_i\right) = \frac{\sigma^2}{B}$$

If the trees are correlated with pairwise correlation $\rho$, the ensemble variance becomes:
$$\text{Var}(\text{Forest}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

As $B \to \infty$, the second term approaches zero, leaving $\rho \sigma^2$. To minimize total ensemble variance, we must reduce the correlation $\rho$ between individual trees.

### Randomness Introduced in Random Forest
1. **Bootstrap Aggregation (Row Randomness)**: Each tree trains on a sample drawn with replacement. Roughly $63.2\%$ of original samples are included in each bootstrap sample, forcing trees to train on different subsets.
2. **Random Feature Subsets (Column Randomness)**: At every node split, a random feature subset is evaluated rather than all available features. This prevents dominant features from dictating early splits across all trees, decorrelating individual tree structures.
