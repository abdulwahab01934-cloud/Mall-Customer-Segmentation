[README.md](https://github.com/user-attachments/files/34281970/README.md)
# Mall Customer Segmentation

An unsupervised machine-learning project that groups mall customers using their annual income and spending score. The goal is to identify interpretable customer segments that can help a mall understand its customer base and support more targeted marketing decisions.

## Group members

- Abdul Wahab
- Govind Raj Choudhary
- Magnus Nilsson
- Noman Ghauri

## Project overview

The notebook uses the Mall Customers dataset and explores customer segmentation with **K-Means** as the primary clustering method. It also compares MiniBatch K-Means, investigates engineered features, uses DBSCAN as an alternative density-based method, and trains a small decision tree to turn the discovered segments into readable assignment rules.

### Main objectives

- Explore the customer dataset and check its quality.
- Identify groups of customers with similar income and spending behaviour.
- Select and interpret a practical number of clusters.
- Compare K-Means with MiniBatch K-Means.
- Test whether additional features improve cluster quality and stability.
- Examine DBSCAN's clusters and noise points.
- Produce understandable rules for assigning new customers to segments.

## Dataset

Place `Mall_Customers.csv` in the same directory as the notebook.

The notebook expects these original columns:

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Genre` | Customer gender |
| `Age` | Customer age in years |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Spending score on a 1–100 scale |

The notebook renames these columns to `customer_id`, `gender`, `age`, `income`, and `spending`.

The dataset shown in the notebook contains **200 customers**. Its initial checks report no missing values and no duplicate customer IDs.

## Methods and workflow

1. **Load and inspect the data** — read the CSV, standardise column names, inspect data types and descriptive statistics, and check missing values, duplicate IDs, and gender counts.
2. **Exploratory data analysis** — plot numerical distributions, pairwise relationships, and correlations.
3. **Prepare clustering features** — use `income` and `spending` for the baseline clustering. The customer ID is an identifier rather than a behavioural feature, and gender is not included in the baseline distance calculation. Standardise the two features with `StandardScaler`.
4. **Choose the number of clusters** — examine the K-Means elbow curve and silhouette scores. The notebook proceeds with **k = 5**.
5. **Fit and interpret K-Means** — train K-Means, inspect cluster centres in the original feature units, and assign descriptive business names based on income and spending:
   - Target customers
   - Careful high earners
   - Enthusiastic spenders
   - Budget-conscious
   - Average customers
6. **Inspect cluster boundaries and stability** — visualise the clusters and decision regions, compare single versus multiple random initialisations, and measure agreement using the Adjusted Rand Index (ARI).
7. **Compare K-Means and MiniBatch K-Means** — compare fitting time, inertia, silhouette score, and ARI agreement with the K-Means result.
8. **Experiment with engineered features** — test spending-to-income ratio, an under-40 indicator, age, and a binary gender feature in several feature sets. Compare silhouette scores with a random-data reference and estimate bootstrap stability using ARI.
9. **Profile the segments** — summarise customer counts, average income, spending, age, spending per income unit, age mix, and gender composition.
10. **Explore DBSCAN** — use a k-distance plot to inform the `eps` choice, fit DBSCAN, identify noise points, compare its assignments with K-Means, and test parameter sensitivity.
11. **Create interpretable assignment rules** — train a depth-limited decision tree using the K-Means-derived segment names as labels. Evaluate it with five-fold stratified cross-validation and predict segments for example new members.

## Algorithms and evaluation

| Method | Role in this project |
|---|---|
| K-Means | Main clustering model; uses `k = 5` for the segment interpretation |
| MiniBatch K-Means | Faster alternative compared with the standard K-Means fit |
| DBSCAN | Exploratory density-based clustering and noise detection |
| Decision Tree Classifier | Learns readable rules to approximate the K-Means segment assignments |

The notebook uses the following measures:

- **Inertia** — within-cluster sum of squared distances; lower values indicate more compact clusters for a given setup.
- **Silhouette score** — measures how well-separated and cohesive clusters are; higher values are generally better.
- **Adjusted Rand Index (ARI)** — measures agreement between two sets of cluster assignments, adjusted for chance.
- **Bootstrap stability (ARI)** — estimates how consistently clustering assignments agree across bootstrap samples.
- **Macro F1** — evaluates the decision tree's segment predictions while giving each segment equal weight.

## Feature sets tested

The feature-engineering experiment compares:

- Baseline: `income`, `spending`
- Baseline plus `ratio` (`spending / income`)
- Baseline plus `young` (1 when age is below 40)
- `age`, `income`, `spending`
- `age`, `income`, `spending`, and `male` (1 for Male, 0 otherwise)

The notebook evaluates candidate cluster counts from 2 to 8 for these feature sets. It also compares each feature set's best silhouette score with a reference generated from random values over the observed feature ranges.

## Requirements

Python 3.9 or later is recommended. Install the libraries used in the notebook:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Run the project

1. Clone or download this repository.
2. Put `Mall_Customers.csv` in the project directory.
3. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
4. Run the cells from top to bottom.

The notebook imports:

```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
```

## Outputs

Running the notebook produces:

- Dataset inspection and descriptive statistics.
- Histograms, scatter plots, and a correlation matrix.
- Elbow and silhouette analyses for selecting `k`.
- K-Means cluster centres and customer segment labels.
- Cluster visualisations with decision boundaries.
- Initialisation-stability and K-Means/MiniBatch K-Means comparisons.
- Feature-engineering comparison against a random-data reference.
- Segment profile tables and visualisations.
- DBSCAN cluster/noise summaries and parameter-sensitivity results.
- Decision-tree rules, cross-validation scores, and example predictions for new members.

## Important interpretation notes

- The five segment names are descriptive labels assigned from the K-Means cluster centres; they are not pre-existing labels in the dataset.
- The decision tree is trained on the segments produced by K-Means. Its cross-validation score measures how well it reproduces those clustering-derived labels, **not** how accurately it predicts an independently observed customer outcome.
- The engineered-feature experiments are comparisons of clustering structure and stability. A higher silhouette score alone does not prove that a feature set is more useful for business decisions.
- DBSCAN is an extension/alternative analysis. Its output depends on `eps` and `min_samples`; noise points are labelled `-1`.
- The notebook's segment profiles use the segment labels created by the baseline K-Means model.

## Project structure

```text
mall-customer-segmentation/
├── Mall_Customers.csv
├── <notebook>.ipynb
└── README.md
```

Replace `<notebook>.ipynb` with the actual notebook filename when adding it to the repository.

## Possible future improvements

- Validate the segments with additional customer or transaction data.
- Investigate whether the segments remain useful on a larger or newer customer sample.
- Compare business usefulness, not only clustering metrics, before selecting a final feature set.
- Save the fitted scaler and clustering model so the same preprocessing and assignments can be applied consistently to future customers.
