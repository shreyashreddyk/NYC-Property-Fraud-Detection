# NYC Property Tax Fraud Detection using Unsupervised Learning

This repository contains an end-to-end data science project focused on identifying potential tax fraud in New York City's property assessment data. Given the absence of labeled data for fraudulent activities, this project leverages **unsupervised anomaly detection** techniques to score over one million property records and surface the most unusual cases for review.

The methodology employs a robust pipeline of data cleaning, extensive feature engineering, and a hybrid scoring system based on two complementary unsupervised models: a distance-based algorithm and a neural network autoencoder.

---

### Project Reports

For a comprehensive understanding of the project's methodology, findings, and technical details, please see the detailed reports included in this repository:

* **[Final Report](./Final_Report.pdf):** A complete summary of the project from data cleaning to the analysis of top anomalies.
* **[Feature Engineering & Modeling](./Feature_Engineering_Unsupervised_Modeling.pdf):** A technical deep-dive into the construction of the 29 analytical variables and the mathematical basis for the two anomaly detection models.
* **[Data Cleaning Report](./Data_Cleaning.pdf):** Details the rigorous process of data imputation and the logic for record exclusion.
* **[Data Quality Report](./DQR.pdf):** An initial exploratory analysis of the raw dataset.

---

## Project Pipeline

The project follows a structured workflow designed to handle large-scale, messy data and extract meaningful, actionable insights without relying on pre-existing labels.

### 1. Data Ingestion and Quality Assessment
* The analysis begins with the NYC property assessment dataset, containing **~1.07 million records** and 32 initial variables.
* A preliminary Data Quality Report (DQR) was generated to understand field types, distributions, and the extent of missing or invalid data.

### 2. Data Cleaning and Preprocessing
A careful cleaning strategy was implemented to prepare the data for modeling while preserving the very anomalies we aim to detect.
* **Outlier Retention:** Unlike typical data pipelines, statistical outliers in financial and dimensional fields were deliberately retained, as they represent potential cases of fraud or assessment error.
* **Exclusion of Irrelevant Records:** Approximately 26,500 records corresponding to government-owned properties, public authorities, and other non-taxable entities were filtered out using a keyword-based exclusion list on the `OWNER` field. These records often have idiosyncratic valuations that would otherwise confuse the anomaly detection algorithms.
* **Hierarchical Missing Data Imputation:** Missing values in nine critical fields (`FULLVAL`, `AVLAND`, `AVTOT`, `ZIP`, `STORIES`, etc.) were imputed using a hierarchical strategy. This involved filling missing data with medians or means from progressively broader groups of similar properties (e.g., by Tax Class + Borough + Building Class, then by Tax Class + Borough, and finally by Tax Class alone). This ensures that imputed values are contextually plausible and do not artificially create new anomalies.

### 3. Feature Engineering
The core of the project's domain expertise is encapsulated in the creation of **29 engineered features**. These features transform raw data into metrics designed to capture valuation inconsistencies.
* **Base Size Features:** `lotarea`, `bldarea`, and `bldvol` were calculated to provide a standardized measure of property size.
* **Value-to-Size Ratios:** Nine fundamental ratios of property value to size were created (e.g., $r_{ij} = V_i / S_j$, such as `FULLVAL/lotarea`). These metrics measure value density.
* **Reciprocal Transformation for Outlier Detection:** To capture both over- and under-valued properties, each ratio $r$ was transformed into $R = \max(r, 1/r)$. This ensures that both extremely high and extremely low value-density properties manifest as high-magnitude outliers.
* **Peer-Comparison Features:** Each of the nine base ratios was normalized against the mean ratio for its **ZIP code** and **Tax Class**. This contextualizes a property's valuation against its geographical and typological peers, isolating true anomalies from neighborhood effects. This step generated 18 new features.
* **Bespoke Ratios:** Two final features were added to check for internal consistency: `AssessRatio` (`FULLVAL`/`AVTOT`) and `BldgLotRatio` (`bldarea`/`lotarea`).

### 4. Anomaly Detection Modeling
Two complementary unsupervised models were trained on the 29 engineered features to generate independent anomaly scores for each property.

#### Model 1: Z-Score Distance in PCA Space
This model identifies points that are far from the center of the data distribution in a decorrelated feature space.
1.  All 29 features were **Z-score standardized**.
2.  **Principal Component Analysis (PCA)** was applied to reduce dimensionality and handle multicollinearity. The first `m` components explaining ~90% of the variance were retained.
3.  The resulting principal components were standardized again to ensure each component was weighted equally.
4.  The anomaly score, $D_i$, was calculated as the **Minkowski distance** ($p=2$, Euclidean) of each property's transformed feature vector from the origin. This score is mathematically equivalent to the Mahalanobis distance if all components are used.
```math
D_{i} = \sqrt{z_{i1}^{2} + z_{i2}^{2} + \dots + z_{im}^{2}}
```

#### Model 2: Autoencoder Neural Network
This model uses a neural network to learn a compressed representation of "normal" data and flags anomalies as those with high reconstruction error.
1.  The same standardized feature set was used as input.
2.  A simple **Autoencoder** with a bottleneck hidden layer (`15 -> 8 -> 15`) was trained to reconstruct its input.
3.  The anomaly score, $E_i$, was defined as the **Mean Squared Error (MSE)** between the original input vector ($z_i$) and the reconstructed output vector ($z'_i$).
```math
E_{i} = ||z_{i} - z'_{i}||^{2} = \sum_{j=1}^{m}(z_{ij} - z'_{ij})^{2}
```

### 5. Ensemble Scoring
The scores from the two models, $D_i$ and $E_i$, were on different scales. To create a final, robust ranking, they were combined:
1.  **Rank-Order Scaling:** Each score was converted to its percentile rank to bring them to a common, non-parametric scale.
2.  **Averaging:** The final anomaly score, $S_i$, was calculated as the simple arithmetic mean of the two rank-scaled scores.

### 6. Analysis of Results
Properties were sorted by the final score $S_i$ in descending order to produce a final investigation list. Manual review of the top-scoring records revealed clear and actionable anomalies, such as:
* An enormous lot with a near-zero market value.
* A Manhattan high-rise with a building footprint 42 times larger than its recorded lot size.
* A $43.1M townhouse in a tax class of single-family homes, with an unusually low assessed value relative to its market value.

These findings, detailed in the final report, confirm the model's effectiveness at surfacing properties with implausible physical or financial characteristics.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/NYC-Property-Tax-Fraud-Detection.git](https://github.com/your-username/NYC-Property-Tax-Fraud-Detection.git)
    cd NYC-Property-Tax-Fraud-Detection
    ```
2.  **Set up a virtual Python environment and install required Python packages. Also setup ipykernel to handle ipynb.** 
    
4.  **Download the data:**
    The dataset can be found on the [NYC Open Data portal](https://data.cityofnewyork.us/Housing-Development/Property-Valuation-and-Assessment-Data/rgy2-tti8). Download the CSV and place it in the `data/` directory.

5.  **Run the notebook:**
    Launch Jupyter Notebook and open `NY_unsupervised_fraud.ipynb`.
    ```bash
    jupyter notebook
    ```

---

## Key Skills Demonstrated

* **Data Science & Analysis:**
    * End-to-End Project Management
    * Exploratory Data Analysis (EDA)
    * Data Quality Assessment
    * Advanced Data Cleaning & Preprocessing
    * Sophisticated Missing Value Imputation (Hierarchical Approach)
    * Domain-Specific Feature Engineering

* **Machine Learning:**
    * Unsupervised Learning
    * Anomaly & Outlier Detection
    * Dimensionality Reduction (PCA)
    * Neural Networks (Autoencoders for Anomaly Detection)
    * Ensemble Methods (Score Combination and Scaling)

* **Technical Tools & Libraries:**
    * **Python:** Pandas, NumPy, Scikit-learn
    * **Data Visualization:** Matplotlib, Seaborn
    * **Development Environment:** Jupyter Notebook
