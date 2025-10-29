# Principal Component Analysis (PCA) — Complete Guide

## Motivation: Curse of Dimensionality

- Adding more features can hurt model performance due to the curse of dimensionality.
- As dimensions increase, data becomes sparse and the number of samples needed grows exponentially.
- Distance-based algorithms lose effectiveness in high dimensions.
- Visualization and intuition break down in high-dimensional spaces.

## Feature Selection vs. Dimensionality Reduction

| Approach                | Keeps Original Features? | Captures Feature Interactions? | Interpretability | Example Methods         |
|-------------------------|-------------------------|-------------------------------|-----------------|------------------------|
| Feature Selection       | Yes                     | No                            | High            | Forward selection, LASSO|
| Dimensionality Reduction| No (creates new ones)   | Yes                           | Lower           | PCA, t-SNE, LDA        |

- Feature selection picks a subset of original features.
- Dimensionality reduction (like PCA) creates new features (components) that capture complex correlations.

## PCA Intuition

- PCA transforms correlated features into uncorrelated principal components.
- Principal components are linear combinations of the original features.
- The first principal component captures the maximum variance; each subsequent component is orthogonal and captures the next highest variance.

## Why Standardize Data Before PCA?

- Features with larger scales dominate principal components if not standardized.
- Standardization (mean=0, std=1) ensures all features contribute equally.

## Mathematical Formulation

PCA seeks the direction **a** that maximizes the variance of the projected data:

$$
\max_{\mathbf{a}} \ \mathbf{a}^T \mathbf{S} \mathbf{a} \quad \text{subject to} \quad \mathbf{a}^T \mathbf{a} = 1
$$

Where:  
- $\mathbf{a}$: Eigenvector (principal component direction)  
- $\mathbf{S}$: Covariance matrix of the data  
- $\mathbf{a}^T \mathbf{S} \mathbf{a}$: Variance of data projected onto $\mathbf{a}$  
- $\mathbf{a}^T \mathbf{a} = 1$: Unit length constraint  

This leads to the eigenvalue equation:

$$
\mathbf{S}\mathbf{a} = \lambda \mathbf{a}
$$

## Interpreting PCA Results

- **Principal components**: New, uncorrelated variables capturing variance.
- **Explained variance ratio**: Shows how much information each component retains.
- **Loadings**: Coefficients showing how much each original feature contributes to each principal component.
- **Dimensionality reduction**: Use first few PCs for further analysis, modeling, or visualization.

## Quiz Questions

1. Why does adding more features sometimes hurt model performance?
2. What is the difference between feature selection and dimensionality reduction?
3. Why is it important to standardize data before applying PCA?
4. What does the eigenvalue equation $ \mathbf{S}\mathbf{a} = \lambda \mathbf{a} $ represent in PCA?
5. How do you decide how many principal components to keep?

## PCA Implementation in Python

```import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Example dataset
data = {
    'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
    'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 29]
}
df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Explained variance ratio
print(pca.explained_variance_ratio_)

# Principal component weights
print(pca.components_)

# Visualize
plt.scatter(principal_df['PC1'], principal_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection')
plt.show()```


