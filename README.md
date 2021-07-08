# Linear_Regression



## Overview

Linear regression models for Jupyter Notebook

## Dataset 

[The Auto MPG dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/)

This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. The dataset was used in the 1983 American Statistical Association Exposition.

Data Set Information:

This dataset is a slightly modified version of the dataset provided in the StatLib library. In line with the use by Ross Quinlan (1993) in predicting the attribute "mpg", 8 of the original instances were removed because they had unknown values for the "mpg" attribute. The original dataset is available in the file "auto-mpg.data-original".

"The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes." (Quinlan, 1993)

Attribute Information:

- **mpg: continuous
- cylinders: multi-valued discrete
- displacement: continuous
- horsepower: continuous
- weight: continuous
- acceleration: continuous
- model year: multi-valued discrete
- origin: multi-valued discrete
- car name: string (unique for each instance)

![dataset](https://user-images.githubusercontent.com/57882064/124872325-bcbe4880-df8a-11eb-8480-250f5ef7534e.png)


## Model

### 1. Data preprocessing

- Clean unknown values (n/a or NA)
- One-Hot Encoding (USA / Europe / Japan)
- Split train / test dataset

### 2. Ordinary least squares (OLS) regression model

```python
import statsmodels.api as sm

X_train = sm.add_constant(X_train) # Constant value added (1.0 added into dataset)
model = sm.OLS(y_train, X_train, axis = 1)
model_trained = model.fit()

```

### 3. Normality and homoscedasticity

#### Residual QQ Plot

- If the resudual is normally distributed, the points in the QQ-normal plot lie on a straight diagonal line
![qqplot](https://user-images.githubusercontent.com/57882064/124878410-b1bae680-df91-11eb-9242-118b3a5425d5.png)

#### Residual Scatter Plot

- Heteroscedasticity (Dependent variable for Raw data)
![Heteroscedasticity](https://user-images.githubusercontent.com/57882064/124874825-ceedb600-df8d-11eb-8e78-a2859b4960a8.png)

- homoscedasticity (Dependent variable -> ln(y))
![homoscedasticity](https://user-images.githubusercontent.com/57882064/124878189-71f3ff00-df91-11eb-968f-47169c2e2741.png)

### 4. 
