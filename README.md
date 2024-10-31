<p align="center">
  <img src="https://github.com/user-attachments/assets/71c254f9-2ffb-4ee2-add3-dd1cbbf00a92" alt="MB" width="2500" height="200">
</p>

## INTRODUCTION: 

Linear and Logistic Regression are two common techniques used to predict outcomes, but they are suited for different types of problems.

### Linear Regression 
Linear Regression is used when we want to predict a continuous value, (e.g the price of a car, temperature, or someone's age). The goal is to fit a straight line through the data, where the input factors (like mileage or engine size) help us estimate the output like the car’s price. It minimizes the difference between actual and predicted values, allowing us to use that line to make predictions for future data.

### Logistic Regression 
Logistic Regression, on the other hand, is used when we need to predict a category, like "yes or no" decisions. For example, it helps to answer questions like "Will it rain tomorrow?" or "Is this email spam?" It doesn’t provide a straight line but instead predicts the probability that something will happen, usually expressed between 0 and 1.

##### The main difference between the two is the kind of data they deal with—linear regression works for continuous outcomes, while logistic regression handles categorical outcomes. Both are key tools in making data-driven decisions, helping us spot trends and predict results based on available information.

### DATASET DESCRIPTION: 
### 1. Car Price Prediction Dataset (Linear Regression)
The dataset presents information on understanding the car features that actually influence the price of cars in various countries. In particular, it is the Chinese Automobile Company Geely Auto. They aspire to enter the US market by setting up their manufacturing unit there and producing cars locally. To achieve this, it is essential to identify and understand the factors influencing car pricing in the American market, as these factors may differ significantly from those in the Chinese market. Through the application of linear regression analysis, we can model car prices based on the available independent variables. This approach we can manipulate the design of the cars, the business strategy etc. to meet certain price levels. And this will be a good way for management to understand the pricing dynamics of a new market. 
### 2. Customer Satisfaction Dataset (Logistic Regression)
The Customer Satisfaction Dataset, designed for logistic regression analysis, consists of 10,616 entries capturing customer feedback on delivery services. Key variables include Delivery Satisfaction, which is rated on a scale from 1 to 5 and reflects the overall delivery experience, and Food Quality Satisfaction, also rated from 1 to 5, indicating customer satisfaction with the quality of food received. The dataset’s target variable is Order Accuracy, a binary indicator of whether the order was accurate, with 0 representing an inaccurate order and 1 indicating an accurate one. This dataset is valuable for using logistic regression to explore and predict order accuracy based on satisfaction ratings, providing insights into the impact of delivery and food quality on customer perception of order accuracy.

### PROJECT OBJECTIVES: 
The objectives of this project are:
1. **Understand the Fundamentals**: 
   - Grasp the theoretical differences between Linear and Logistic Regression.
   - Identify scenarios where each model is most applicable.
2. **Build Predictive Models**:
   - Apply **Linear Regression** to predict car prices based on the independents given.
   - Use **Logistic Regression** to classify order accuracy based on customer satisfaction
   metrics.
3. **Evaluate Model Performance**:
   - For Linear Regression: Evaluate with metrics like r2 and adjusted r2.
   - For Logistic Regression: Evaluate with metrics like accuracy, precision, recall, and f1-
     score. 
4. **Interpret Model Results**:
   - Analyze how car attributes influence price.
   - Understand how satisfaction metrics impact the likelihood of an accurate order.
     
#### DATA EXPLORATION: 
This phase involves organizing datasets, cleaning them to address missing values and outliers, and eliminating variables that may introduce noise or multicollinearity. The process includes defining independent (input) and dependent (output) variables, encoding categorical data, and applying feature scaling for standardization. The dataset is then split into training (70-80%) and test sets (20-30%) to facilitate model fitting and evaluation. This structured approach enables the application of linear or logistic regression models, with tuning and testing to achieve accurate predictive outcomes.

##### For Linear Regression (CAR PRICING ASSIGNMENT)

###### (Y= CarPrice; dependent variable) 
###### independent variables:
wheelbase (distance between front and rear wheels)
carlength (length of the car)
carwidth (width of the car)
carheight (height of the car)
curbweight (weight of the car without passengers or luggage)
enginesize (size of the engine)
bore (bore ratio)
stroke (stroke volume inside the engine)
compressionratio (compression ratio of the engine)
horsepower (power output of the engine)
peakrpm (maximum engine RPM)
citympg (mileage in city driving)
highwaympg (mileage in highway driving)


Car Prices Data Dictionary 

 <img src="https://github.com/user-attachments/assets/9e010651-7c47-4df2-8f1b-e93f6134bfe0" alt="Screenshot" width="500" height="250">
 
Car_Price Datasheet

<img src="https://github.com/user-attachments/assets/7fb8bcd9-0c7b-4b23-9673-65dfd8f87838" alt="Screenshot" width="500" height="250">

##### Car_Price Datasheet (Cleaned)
-The image below shows the edited datasheet where categorical variables(based on the data dictionary) have been removed for a cleaner and more effective analysis implementation.

<img src="https://github.com/user-attachments/assets/9a8b1f9a-d9c6-42e3-845a-44c7e51e5403" alt="Screenshot" width="500" height="250">

### DATA PREPROCECSSING

Data preprocessing is the process of preparing raw data for analysis and modeling by transforming it into a clean, structured, and standardized format. 

##### Steps in Data Preprocessing (as shown in the image below):

#### IMPORTING LIBRARIES AND DATA SET:
* The code first imports the pandas library, which is commonly used in Python for data manipulation and analysis.
* The dataset, stored in an Excel file (CarPrice_Ass.xlsx), is then loaded into a pandas DataFrame, making it easier to explore and manipulate.

###### Initial Exploration:
* The dataset.head(10) function displays the first 10 rows of the dataset. This initial look helps to understand the structure of the data, the features (columns), and the first few values in each feature.

###### Understanding Features and Target:
* The dataset contains various car attributes (features), such as wheelbase, carlength, curbweight, enginesize, horsepower, etc., along with price, which seems to be the target variable we need to predict.

<img src="https://github.com/user-attachments/assets/cab4ef5e-9aae-4acd-a561-8b3f76168681" alt="Screenshot" width="500" height="250">

#### GETTING INPUT AND OUTPUT
It is the process for selecting the input (independent variables) and output (dependent variable) from the dataset.

##### This is the step-by-step explanation for getting the inputs and output. (as shown in the image below)

###### Selecting Inputs (Independent Variables):

* The line X = dataset.iloc[:, :-1].values selects all rows (:) and all columns except the last one (:-1) from the dataset.
* Here, dataset.iloc[:, :-1] uses the .iloc function, which allows for selection by index position. By specifying :-1, it includes all columns except the last one.
* .values converts the resulting selection into a NumPy array.
* This array X will contain all the features (independent variables) used for the model, excluding the target variable.

###### Selecting Output (Dependent Variable):

* The line y = dataset.iloc[:, -1].values selects all rows (:) and only the last column (-1) from the dataset.
* dataset.iloc[:, -1] refers to the last column, which is assumed to be the target variable (price) in this case.
* .values again converts this selection into a NumPy array.
* This array y contains the values of the dependent variable, which is the car price that the model will try to predict.
  
##### Summary
###### X (input),
includes all features except the target variable (price).
###### y (output),
includes only the price values, which is the variable we're aiming to predict with a regression model.

<img src="https://github.com/user-attachments/assets/64aea8f5-e456-4e0a-b939-a21e120e7710" alt="Screenshot" width="500" height="250">

<img src="https://github.com/user-attachments/assets/edef7cc1-45de-4299-b846-2a8d3c529285" alt="Screenshot" width="500" height="250">

#### CREATING TRAINING AND TEST SET

 It is the process of splitting the dataset into training and test sets, which is a common step in preparing data for machine learning. 

###### Explanation of Code and Outputs (based on the figure below)

###### Importing train_test_split:

* The code starts by importing the train_test_split function from sklearn.model_selection.
* This function is designed to split the dataset into separate training and testing subsets for both the independent (input) and dependent (output) variables.

###### Splitting the Data:

* The line X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) is used to split the X and y arrays into training and test sets.
* X represents the input features (independent variables).
* y represents the target variable (dependent variable).
* test_size=0.2 specifies that 20% of the data should be used for testing, while the remaining 80% will be used for training.
* random_state=0 is a seed value that ensures the split is reproducible (the same split occurs every time the code is run).
  
###### Resulting Variables:
* X_train and y_train: These arrays contain 80% of the data, which will be used to train the model.
* X_test and y_test: These arrays contain the remaining 20% of the data, which will be used to evaluate the model's performance.

###### Example Output of Split Data
* X_train: This is the part of the X array that is used for training the model. The content is a subset of the original data, containing 80% of the rows.
* X_test: This is the part of the X array used for testing the model, containing the remaining 20% of rows.

###### Similarly:

* y_train: This would be the corresponding 80% subset of the y array (target variable values) used for training.
* y_test: This is the corresponding 20% subset of the y array used for testing.
  
##### Summary
This process helps create separate training and test datasets:

* Training Set (X_train, y_train): Used to fit the model, allowing it to learn patterns in the data.
* Test Set (X_test, y_test): Used to evaluate how well the model generalizes to unseen data.

<p align="left">
  <img src="https://github.com/user-attachments/assets/0b5954a6-4912-418f-a9a0-a9369b43950a" width="700" height="300" alt="training and test set-VEED">
</p>

#### MODEL BUILDING AND TRAINING : 
Developing linear regression approach that will predict a continuous or numerical outcome, such as car price, horsepower, and so on (dagdagan ng about logistic). This involves combining various methods and analytical techniques to achieve a more accurate and successful prediction of the outcome.

##### For Linear Regression 

### MODEL IMPLEMENTATION

#### Building and Training the Model 
This is where a linear regression model is being built, trained, and used for inference.

##### Here's an explanation of the steps (based on the shown figure below) :

###### 1. Building the Model: 
From the code below, imports the LinearRegression class from Scikit-learn's linear_model module. This library provides efficient machine learning algorithms and tools, and LinearRegression is a core component for building linear regression models.

from sklearn.linear_model import LinearRegression
model = LinearRegression()

* This code imports the LinearRegression class from the sklearn.linear_model module (part of the popular Scikit-learn library).
* Then, an instance of the LinearRegression() model is created and stored in the model variable. This prepares the linear regression model, which will later be trained on data.

###### 2. Training the Model:
model.fit(X_train, y_train)

* In this step, the linear regression model (model) is trained using the fit() method.
* X_train is the input data (features) used to train the model, and y_train is the target output (labels).
* The fit() method estimates the best-fitting line (in case of one feature) or hyperplane (if there are multiple features) for the training data.

###### 3. Interference
y_pred = model.predict(X_test)
y_pred

* Making Predictions:

- The predict() method is used to generate predictions based on the unseen test data (X_test). This test dataset contains new data points for which the model hasn’t seen the target variable.
- The variable y_pred stores the predicted values for X_test. These are the output (dependent variable) estimates that the model believes to be correct based on the relationships it learned during training.

* Predicted Output:

- The output shows an array of predicted values for the test data points. These values represent the model’s estimations based on the input features from X_test. For instance:
[6685.3704445, 18685.04379745, 15824.2008859, 129.99461763, ...]
These numbers represents car prices of the dataset.

##### Inference with a Single Data Point:
model.predict([[1,88.6,168.8,64.1,48.8,2548,130,3.47,2.68,9,111,5000,21,27]])

* Predicting for a Single Data Point:

- This line makes a prediction using a specific, single instance of data (a row of features).
- The predict() method takes in a list of features that match the structure of the training data. In this case, a list is passed with values like:
  [1, 88.6, 168.8, 64.1, 48.8, 2548, 130, 3.47, 2.68, 9, 111, 5000, 21, 27]
- These numbers are most likely the feature values for various characteristics that the model will use to predict the target variable (car price).

* Predicted Value for the Single Data Point:
The result of this prediction is shown as: array([14628.18813458])
* This means that based on the input features provided, the model predicts a value of approximately 14628.19 for the target variable (car price).

<p align="left">
  <img src="https://github.com/user-attachments/assets/485a0190-f054-4e8e-a6c2-571299437bbe" width="700" height="300" alt="Building and training model-VEED">
</p>

##### I observed that the interference of single data points did not align closely with the model's fit, so I repeated the analysis, removing the integer variable, Car_Id, which I believe may have been interfering with the prediction accuracy.

<p align="left">
  <img src="https://github.com/user-attachments/assets/5fb08741-fbf3-409b-92dc-b134b4c89720" width="700" height="300" alt="PART2-VEED">
</p>


###### Summary:
* The first part of the inference shows predictions made for multiple data points (stored in X_test) using the trained linear regression model.
* The second part shows how you can make a prediction for a single, manually entered data point by passing a list of values to model.predict().

#### MODEL EVALUATION: 
To evaluate the performance of both linear and logistic regression models, appropriate metrics must be used. For linear regression, key metrics include R-squared and Adjusted R-squared and metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), which measure the model’s accuracy in predicting continuous values. (dagdagan ng about sa logistic). These metrics provide a clear view of model reliability, enhancing predictions in both continuous and categorical contexts.

##### For Linear Regression 

### EVALUATION METRICS

The model evaluation uses two key metrics: R-Squared (R²) and Adjusted R-Squared.

1. R-Squared (R²): The R² score, calculated here as approximately 0.8455, represents the proportion of the variance in the dependent variable (car price) that is predictable from the independent variables. A higher R² value (close to 1) indicates a better fit, meaning that the model explains a significant portion of the variance in car prices based on the provided features.

2. Adjusted R-Squared: Adjusted R², shown as approximately 0.7624, modifies the R² value by penalizing the model for including irrelevant features that do not improve the prediction. This value adjusts for the number of predictors used and is typically lower than R², especially when the model has multiple features. It provides a more accurate measure of model performance when there are multiple predictors.

###### Summary
The model evaluation shows that the model performs reasonably well, as indicated by the high R² and Adjusted R² values. The Adjusted R² being slightly lower than R² suggests that not all features may contribute significantly to the model's predictive accuracy, but overall, the model captures a large portion of the variance in car prices.

<p align="left">
  <img src="https://github.com/user-attachments/assets/37edc2f0-783c-4e3c-908d-39b517a23311" width="700" height="300" alt="Evaluating the model-VEED">
</p>

##### Result of 2nd Analysis
<p align="left">
  <img src="https://github.com/user-attachments/assets/3f68bad8-1fcd-430b-9e00-ec35cc33a4da" width="700" height="300" alt="PART 3-VEED">
</p>

1. The R² value here is approximately 0.818, meaning about 81.8% of the variance in the target variable can be explained by the model’s predictors.
2. The Adjusted R² result here is approximately 0.731, meaning about 73.1% modifies the R² value by penalizing the model for including irrelevant features that do not improve the prediction.

###### Comparison and Interpretation
* It is possible for R² to be higher than Adjusted R², especially when the number of predictors (k) increases. Adjusted R² penalizes R² based on the number of predictors to prevent overfitting, which is why it’s often slightly lower than R² when there are multiple predictors.
* In this example, the difference between R² (0.818) and Adjusted R² (0.731) indicates that while the model explains a good amount of variance, the Adjusted R² suggests some predictors may not be contributing significant information, hence the penalty.

### INTERPRETATION 

#### SIGNIFICANCE OF CO-EFFICIENTS 
In a linear regression model for predicting car prices, the significance of each coefficient reveals the strength and direction of the relationship between each independent variable and the car price (dependent variable). Here's a breakdown of how each independent variable's coefficient might be interpreted in this context:

1. Wheelbase: A positive coefficient for wheelbase suggests that a longer distance between the front and rear wheels is associated with a higher car price, potentially due to the perception of a larger, more stable vehicle.

2. Carlength: If carlength has a significant positive coefficient, longer cars may be more expensive, possibly due to additional materials or the perception of luxury.

3. Carwidth: A positive coefficient for carwidth could indicate that wider cars, which may offer more interior space or stability, tend to be priced higher.

4. Carheight: A significant positive coefficient here might suggest that taller cars, which may offer more headroom or a more robust design, contribute to a higher car price.

5. Curbweight: This variable represents the car's weight without passengers or cargo. A positive coefficient for curbweight implies that heavier cars, often linked with larger or more luxurious models, are generally priced higher.

6. Enginesize: If enginesize has a positive coefficient, larger engines are associated with higher prices, likely due to the increased power output and associated manufacturing costs.

7. Bore: Bore ratio affects the engine’s performance. A significant positive coefficient indicates that a larger bore, which can increase engine performance, tends to drive up the car’s price.

8. Stroke: Stroke volume within the engine affects performance and efficiency. A positive coefficient would imply that a larger stroke is associated with a higher price, possibly due to performance benefits.

9. Compressionratio: If this has a significant coefficient, it indicates that a higher compression ratio, which affects engine efficiency, has an impact on price. A positive value could suggest that cars with more efficient or powerful engines are priced higher.

10. Horsepower: A positive coefficient for horsepower would mean that cars with greater power output are typically more expensive, reflecting the cost and demand for powerful engines.

11. Peakrpm: This variable represents the maximum RPM the engine can reach. A significant coefficient here would indicate that higher peak RPMs have a measurable impact on price, potentially tied to the car’s performance capabilities.

12. Citympg and Highwaympg: These variables represent fuel efficiency in city and highway driving conditions. If these have significant positive coefficients, they suggest that more fuel-efficient cars (higher mpg) are more expensive. Conversely, negative coefficients could imply that higher efficiency correlates with lower prices, possibly reflecting lower production costs or targeting of cost-conscious buyers.

Each coefficient’s significance level (p-value) is crucial to determine whether the variable’s relationship with car price is statistically meaningful. Coefficients with low p-values (typically below 0.05) indicate a statistically significant relationship, suggesting that the variable is likely a true contributor to variations in car prices. The size and sign of the coefficients indicate how much and in what direction each factor influences car price, enabling us to identify key drivers of pricing.

#### MODEL'S PREDICTIVE POWER
The model's predictive power is measured primarily by metrics like R-squared (R²) and Adjusted R-squared. R² indicates the percentage of variance in car prices explained by the independent variables, with a high R² value, close to 1, suggesting that the model captures most of the variability in car prices, demonstrating strong predictive power. However, if R² is low, this suggests that other unmeasured factors may influence price.

Adjusted R-squared refines this measure by penalizing the inclusion of irrelevant predictors. Unlike R², which can increase simply by adding more predictors, Adjusted R² decreases if a predictor doesn’t add meaningful value, helping to prevent overestimation of the model’s power. Together, R² and Adjusted R-squared provide insight into the model's effectiveness at capturing variability, offering a comprehensive view of its predictive strength.

### LOGISTIC REGRESSION (CUSTOMER SATISFACTION ANALYSIS)

##### Dependent variable (y):
Order accuracy
##### Independent variables (X):
Delivery speed satisfaction, 
overall delivery satisfaction, and 
food quality satisfaction

##### Customer Satisfaction Data  
The dataset includes the following columns:
* Customer: Likely a unique identifier for each entry.
* Overall Delivery Satisfaction (1-5 scale): Satisfaction with the delivery experience.
* Food Quality Satisfaction (1-5 scale): Satisfaction with the food quality.
* Delivery Speed Satisfaction (1-5 scale): Satisfaction with delivery speed.
* Order Accuracy (1 = Yes, 0 = No): Whether the order was accurate.
Preview
![image](https://github.com/user-attachments/assets/c9e27966-3cea-4dc1-9578-890b8ea85b5d)

#### METHODOLOGY: Documented steps taken during analysis.

##### Part 1: Data Preprocessing
Data preprocessing is the process of preparing raw data for analysis and modeling by transforming it into a clean, structured, and standardized format. 

###### 1. Importing and Loading the Dataset:
* The code first imports the pandas library, which is commonly used in Python for data manipulation and analysis.
* The dataset, stored in an Excel file (data.csv), is then loaded into a pandas DataFrame called dataset, making it easier to explore and manipulate.
![image](https://github.com/user-attachments/assets/9b1ab6a3-7892-4bc6-b0f9-769d7dbacdf0)
* Creates a copy of the original dataset, named data_filled, which will be used for data cleaning.
![image](https://github.com/user-attachments/assets/c9205f63-f1a5-40f0-ba9d-8e18b34ccd9c)

###### 2. Handling Missing Values:
* Numeric Columns: Missing values in columns 1 to 3 are filled with the mean of each column, a common approach to maintain data consistency for numeric features.
![image](https://github.com/user-attachments/assets/dc84f4ee-753a-49c4-aefb-06a332f37bd8)
* Categorical Column: For the column “Was your order accurate?,” missing values are filled with the mode (most frequent value). This step ensures that categorical data has no null entries, which is essential for model training.
![image](https://github.com/user-attachments/assets/690665ac-a1e3-4ce6-a601-ef0ab04d2519)

###### 3. Verification of Missing Data:
* Prints the count of missing values in each column to ensure that they have been filled.
* isna().sum() is used to print the number of missing values in each column, allowing us to verify that all missing values have been filled.
![image](https://github.com/user-attachments/assets/f45b7f71-2801-4133-8d59-a25b16f85dc2)

###### 4. Encoding the target column as binary values:
* The target column, “Was your order accurate?,” is encoded into binary values: Yes becomes 1, and No becomes 0. Encoding is necessary as logistic regression requires numerical values rather than text labels.
![image](https://github.com/user-attachments/assets/aefea16e-329a-4a5b-a5ee-b6d4309c759e)

###### 5. Preview Data:
* The dataset.head(10) function displays the first 10 rows of the dataset. This initial look helps to understand the structure of the data, the features (columns), and the first few values in each feature.
![image](https://github.com/user-attachments/assets/2ff50cd7-da26-44ec-b763-cd123d836177)

###### 6. Check Data Info:
* Provides summary information about the dataset, including column names, data types, and memory usage.
![image](https://github.com/user-attachments/assets/314d099d-0d03-45e4-ba8a-95da1ee29bcc)
![image](https://github.com/user-attachments/assets/b093b0c5-4ef8-43b1-850b-4420455ad631)
![image](https://github.com/user-attachments/assets/bb9dcc4a-acbd-4cc7-886c-eca5c748ef18)
![image](https://github.com/user-attachments/assets/64389050-1274-4e3d-90a7-17968d7e4690)
![image](https://github.com/user-attachments/assets/97a528c9-ed19-4c4a-ac32-6f9ea9978762)

###### 7. Getting inputs and output:
* Features (X) and the target variable (y) are separated. This is essential for building and training the model, where X represents input variables and y is the output variable the model will predict.
![image](https://github.com/user-attachments/assets/8941238e-f86b-4ecf-946d-e93ef0f1e16a)

###### 8. Class Distribution Check:
* The distribution of target classes is checked to identify any imbalance. Class imbalance can negatively affect model performance by biasing predictions towards the majority class.
![image](https://github.com/user-attachments/assets/b830872e-c626-485d-b382-72e69a74a27e)

###### 9. Data Splitting and Resampling:
* The data is split into training and test sets using train_test_split, with 30% allocated for testing and stratification to preserve the class distribution.
* SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training set to handle class imbalance. SMOTE creates synthetic samples of the minority class, balancing the class distribution and helping the model learn features of both classes more effectively.
![image](https://github.com/user-attachments/assets/60557a56-c55b-4603-9290-31271674dcb9)

##### Part 2: Building and Training the Model
This part involves selecting and training the machine learning model. Here, we initialize a logistic regression model and train it using the preprocessed and balanced dataset.

###### 1. Logistic Regression Model Initialization
* A logistic regression model is initialized with liblinear as the solver, suitable for smaller datasets or binary classification tasks.
![image](https://github.com/user-attachments/assets/ae86e949-1c3c-4309-9086-aff1f2d3fdea)

###### 2. Model Training
* The model is trained using the resampled training data (X_train_res and y_train_res). During this step, the model learns relationships between the input features (X_train_res) and the target variable (y_train_res).
Part 3: Evaluating the Model
![image](https://github.com/user-attachments/assets/761fc471-5c95-491e-b6c2-9168eb39e385)

##### Part 3 - Evaluating the Model
In this stage, we assess the model's performance to understand how well it can predict the target variable. Evaluation metrics help us interpret accuracy, balance, and reliability.

###### 1. Generate Predictions and Classification Report
* Predictions are made on the test set (y_pred).
* A classification report is printed to assess the model’s performance using metrics like:
  Precision: The proportion of true positive predictions among all positive predictions.
  Recall: The proportion of true positives among all actual positives.
  F1 Score: The harmonic mean of precision and recall, which balances the two metrics.
![image](https://github.com/user-attachments/assets/38d34eff-9914-4471-9f2a-3213432f0ec7)

###### 2. Confusion Matrix
* The confusion matrix is printed to show counts of true positives, true negatives, false positives, and false negatives. This helps understand where the model is making errors and if there’s any bias toward one class. A Confusion Matrix Display is also provided.
![image](https://github.com/user-attachments/assets/f92822e8-af7f-4e80-ad58-fbb97efcc17a)
![image](https://github.com/user-attachments/assets/ed2749a2-7274-4fe3-ab34-4fb02b915678)

###### 3. Accuracy Calculation
* Manual Calculation: The accuracy is manually calculated using specific values from the confusion matrix.
![image](https://github.com/user-attachments/assets/8d668a4b-7d52-4da9-95a0-6824588f8520)
* Alternative Calculation Using accuracy_score: For verification, accuracy is also calculated using the accuracy_score function from sklearn, which provides a quick way to confirm the accuracy score.
![image](https://github.com/user-attachments/assets/aed07101-e3f6-48eb-90bf-7e5398aa1f02)


#### RESULTS: Summarized findings
Based on the Methodology, here’s a summary of the findings and model results:

###### 1. Data Preprocessing:
* Missing values were successfully filled using the mean for numeric columns and the mode for the categorical target column.
* The target variable was encoded into binary values (1 for "Yes" and 0 for "No"), making it suitable for logistic regression.
* Class imbalance was observed and addressed with SMOTE, ensuring that both classes were represented equally in the training data.

###### 2. Model Training:
* A logistic regression model was built and trained on the resampled (balanced) dataset. This model aims to predict the accuracy of orders based on input features, which were split into training and test sets.

###### 3. Model Evaluation:
* The classification report indicated metrics such as precision, recall, and F1-score for both classes.
  * Precision: Measures the proportion of true positive predictions to the total positive predictions (both true and false positives).
      * For class "0": 0.28 (low precision, indicating many false positives)
      * For class "1": 0.75 (high precision, meaning fewer false positives)
  * Recall: Measures the proportion of true positive predictions out of all actual positives.
      * For class "0": 0.53 (only about half of actual "0" instances are correctly identified)
      * For class "1": 0.50 (half of the "1" instances are correctly identified)
  * F1-Score: The harmonic mean of precision and recall, balancing both metrics. A higher F1 score suggests better balance.
      * For class "0": 0.37
      * For class "1": 0.60
  * Macro Average: An unweighted average of precision, recall, and F1 scores across classes.
    * Precision: 0.51
    * Recall: 0.52
    * F1 Score: 0.49
  * Weighted Average: A weighted average of the metrics across classes, taking into account the support (number of instances) for each class.
    * Precision: 0.62
    * Recall: 0.51
    * F1 Score: 0.54
* The confusion matrix provided a detailed breakdown of true positives, true negatives, false positives, and false negatives, showing how well the model predicted each class.
  * Structure of the Confusion Matrix
      * True Label: Rows represent the actual labels (ground-truth values) in the dataset.
        * Row 0 represents instances that are actually class "0."
        * Row 1 represents instances that are actually class "1."
      * Predicted Label: Columns represent the model’s predictions.
        * Column 0 represents instances predicted as class "0."
        * Column 1 represents instances predicted as class "1."
  * Values in the Matrix
      * True Negatives (TN) (Top-left: 456):
        * These are cases where the actual label is "0," and the model correctly predicted "0."
        * There are 456 true negative predictions.
      * False Positives (FP) (Top-right: 398):
        * These are cases where the actual label is "0," but the model incorrectly predicted "1."
        * There are 398 false positive predictions.
      * False Negatives (FN) (Bottom-left: 1156):
        * These are cases where the actual label is "1," but the model incorrectly predicted "0."
        * There are 1156 false negative predictions.
      * True Positives (TP) (Bottom-right: 1175):
        * These are cases where the actual label is "1," and the model correctly predicted "1."
        * There are 1175 true positive predictions.
* The accuracy score (calculated manually and verified using accuracy_score) summarized the model's overall performance, indicating how many predictions were correct out of the total predictions made.
  * Overall Accuracy: The proportion of correct predictions for the entire dataset, which is 0.51 or 51%.
     
#### DISCUSSION: Reflection from the results and the encountered limitations.
The results suggests that the model is not effectively learning to distinguish between the two classes. Although it’s making some correct predictions for each class, the number of incorrect predictions (both false positives and false negatives) is quite high. This points to several potential issues:
* High Misclassification Rates:
  * There are 398 false positives and 1156 false negatives, which means the model is frequently confusing one class for the other.
  * This level of misclassification indicates that the model struggles to reliably learn and apply the features that differentiate class "0" from class "1."
* Low Recall for Class "1":
  * The large number of false negatives (1156) compared to true positives (1175) shows that the model fails to identify many actual instances of class "1."
  * Low recall for class "1" suggests the model is not learning the key patterns for accurately identifying this class.
* Overfitting or Underfitting:
  * The model could be underfitting, meaning it hasn’t learned enough to make reliable predictions for either class.
  * Alternatively, it could be overfitting to specific patterns in the training data that don’t generalize well to new data, leading to poor performance.
In summary, while the model makes correct predictions for both classes, the high error rates show that it’s not effectively learning the underlying distinctions between the classes. This indicates a need for improvement, possibly through tuning the model, selecting different features, gathering more data, or trying a different algorithm.

#### Interpretation: Model's ability to classify and the importance of features.
  Upon balancing the classes of the dataset, the accuracy results to 51%. This accuracy is close to random guessing for a binary classification problem, suggesting that the model struggles to find patterns linking the features (satisfaction ratings) to the target variable (order accuracy). In which it suggests that the problem might be:
  * Weak Predictive Power of Features:
    * The satisfaction ratings for delivery experience, food quality, and speed likely have limited predictive power regarding whether an order will be accurate.
    * This might imply that order accuracy depends more on other operational factors, such as order preparation or logistical issues, rather than customer satisfaction metrics alone.
  * Class Balance and Model Complexity:
    * After balancing, the model does not favor either class and should ideally improve recall for previously underrepresented classes (e.g., inaccurate orders).
    * However, with an accuracy close to random (51%), the model may be unable to effectively differentiate between accurate and inaccurate orders based on the given features.
  *  Need for Additional Features:
    *  To enhance predictive performance, additional features—such as specific details about the delivery process, item complexity, or even environmental factors (e.g., weather, peak times)—might be necessary to provide the model with a more meaningful context for predicting order accuracy.
     
#### REAL-WORLD APPLICATION
The knowledge gained from building and evaluating regression models will be applied to real-world scenarios. Linear regression can be used to predict numerical outcomes such as stock prices, real estate values, and energy consumption. Logistic regression will assist in solving classification problems such as predicting customer churn, detecting fraud, or diagnosing medical conditions based on patient data and so on. By interpreting the outcomes of these models, organizations can make informed decisions, optimize operations, and better understand the factors influencing the predictions.
