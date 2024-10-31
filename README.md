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
   - Apply **Linear Regression** to predict car prices based on engine performance and fuel
   efficiency.
   - Use **Logistic Regression** to classify order accuracy based on customer satisfaction
   metrics.
3. **Evaluate Model Performance**:
   - For Linear Regression: 
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

##### Data Preprocessing at VSCode

Data preprocessing is the process of preparing raw data for analysis and modeling by transforming it into a clean, structured, and standardized format. 

##### Steps in Data Preprocessing (as shown in the image below):

###### Importing Libraries and Dataset:
* The code first imports the pandas library, which is commonly used in Python for data manipulation and analysis.
* The dataset, stored in an Excel file (CarPrice_Ass.xlsx), is then loaded into a pandas DataFrame, making it easier to explore and manipulate.

###### Initial Exploration:
* The dataset.head(10) function displays the first 10 rows of the dataset. This initial look helps to understand the structure of the data, the features (columns), and the first few values in each feature.

###### Understanding Features and Target:
* The dataset contains various car attributes (features), such as wheelbase, carlength, curbweight, enginesize, horsepower, etc., along with price, which seems to be the target variable we need to predict.

<img src="https://github.com/user-attachments/assets/cab4ef5e-9aae-4acd-a561-8b3f76168681" alt="Screenshot" width="500" height="250">

##### Getting input and output
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

##### Creating the Training and Test set

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

##### Building and Training the Model 
This is where a linear regression model is being built, trained, and used for inference.

##### Here's an explanation of the steps (based on the shown figure below) :

###### 1. Building the Model:
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

##### Evaluating the Model 
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

#### REAL-WORLD APPLICATION: 
The knowledge gained from building and evaluating regression models will be applied to real-world scenarios. Linear regression can be used to predict numerical outcomes such as stock prices, real estate values, and energy consumption. Logistic regression will assist in solving classification problems such as predicting customer churn, detecting fraud, or diagnosing medical conditions based on patient data and so on. By interpreting the outcomes of these models, organizations can make informed decisions, optimize operations, and better understand the factors influencing the predictions.


