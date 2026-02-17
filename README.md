# Ciencia y Analitica de Datos

Coursework repository for the **Data Science and Analytics** concentration within the Master in Applied Sciences (MNA) program at Tecnologico de Monterrey.

Covers the full data science pipeline: data manipulation, exploratory analysis, visualization, preprocessing, feature engineering, and machine learning -- supported by mathematical foundations in linear algebra, calculus, probability, and statistics.


## Repository Structure

```
Ciencia_Analitica_Datos/
|
|-- Modules/
|   |-- 02_module/   Data Manipulation with Pandas
|   |-- 03_module/   Exploratory Data Analysis and Preprocessing
|   |-- 04_module/   Machine Learning
|
|-- Estadistica/     Statistics and Mathematical Foundations
|
|-- Sesiones_Sincronicas/   Live Session Activities
```


## Module Breakdown

### Module 02 -- Data Manipulation

Hands-on work with pandas for real-world data handling.

- Pandas fundamentals (indexing, filtering, grouping, aggregation)
- Reading and writing data (CSV, JSON, XML)
- Merging, joining, and combining datasets
- Interacting with relational databases (SQL via SQLAlchemy)

Activities: weather data analysis, database manipulation workflows.

### Module 03 -- Exploratory Data Analysis and Preprocessing

End-to-end EDA and data preparation techniques.

- Descriptive statistics and correlation analysis
- Data visualization with matplotlib and seaborn
- Feature scaling and normalization (StandardScaler, MinMaxScaler, RobustScaler)
- Discretization and binning
- Encoding categorical variables (ordinal, one-hot)
- Numerical variable transformations

Activities: EDA on personal loan data, heart disease preprocessing, feature engineering on computer pricing data.

### Module 04 -- Machine Learning

Supervised and unsupervised learning fundamentals.

- ML pipelines and linear regression
- Polynomial regression
- Logistic regression (Titanic survival prediction)
- Principal Component Analysis (PCA) for dimensionality reduction
- Unsupervised learning fundamentals

Activities: PCA on automobile data, linear regression modeling.

### Estadistica -- Mathematical Foundations

Reference materials covering the mathematical backbone of machine learning.

- Linear algebra (vectors, matrices, tensors, eigendecomposition)
- Calculus (derivatives, partial derivatives, integrals, chain rule)
- Probability and information theory
- Statistics (distributions, hypothesis testing)
- Optimization (gradient descent, SGD, learning rate scheduling)
- Bayes' Theorem for data science applications

Includes Jon Krohn's ML Foundations notebooks and PyTorch/TensorFlow examples.

### Sesiones Sincronicas -- Live Sessions

In-class activities with real datasets:

- Forbes 2000 top companies analysis
- Uber trip data exploration
- Fitness data analysis


## Tech Stack

- **Language:** Python 3.12+
- **Data:** pandas, NumPy, SciPy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Database:** SQLAlchemy, PyMySQL
- **Deep Learning (foundations):** PyTorch, TensorFlow
- **Environment:** Poetry, Jupyter Notebooks


## Setup

```bash
# Clone the repository
git clone https://github.com/carloshgalvan95/Ciencia_Analitica_Datos.git
cd Ciencia_Analitica_Datos

# Install dependencies with Poetry
poetry install

# Launch Jupyter
poetry run jupyter notebook
```

Requires Python >= 3.12 and Poetry installed.


## Datasets

The repository includes 35+ datasets used across modules:

- **Module 02:** Adult census, weather, US state demographics, CSV/JSON examples
- **Module 03:** Titanic, wine quality, sonar, breast cancer, personal loan, heart disease, computer prices
- **Module 04:** Housing, automobile, Titanic (logistic regression variant)
- **Sessions:** Forbes 2000, Uber trips, fitness metrics


## License

MIT
