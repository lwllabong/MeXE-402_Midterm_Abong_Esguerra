<p align="center">
  <img src="https://github.com/user-attachments/assets/71c254f9-2ffb-4ee2-add3-dd1cbbf00a92" alt="MB" width="2500" height="200">
</p>

## INTRODUCTION: 

Linear and Logistic Regression are two common techniques used to predict outcomes, but they are suited for different types of problems.

### Linear Regression 
Linear Regression is used when we want to predict a continuous value, (e.g the price of a car, temperature, or someone's age). The goal is to fit a straight line through the data, where the input factors (like mileage or engine size) help us estimate the output like the car’s price. It minimizes the difference between actual and predicted values, allowing us to use that line to make predictions for future data.
 <img src="https://github.com/user-attachments/assets/fb81331d-5981-48e2-8ee9-6434063ea3de" alt="Screenshot" width="800" height="350">
<p align="center">
Linear Regression Example; GIF courtesy: Simplilearn
</p>
<img src="https://github.com/user-attachments/assets/cf6be845-444a-4ae3-abcb-2813f5bfe6de" width="500" height="150">


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

Data preprocessing is the process of preparing raw data for analysis and modeling by transforming it into a clean, structured, and standardized format. In the context of the image you shared, it looks like the initial steps of data preprocessing involve importing and inspecting a dataset of this car price assignment attributes to make it ready for further analysis.

Steps in Data Preprocessing (as shown in the image below):

###### Importing Libraries and Dataset:
* The code first imports the pandas library, which is commonly used in Python for data manipulation and analysis.
* The dataset, stored in an Excel file (CarPrice_Ass.xlsx), is then loaded into a pandas DataFrame, making it easier to explore and manipulate.

###### Initial Exploration:
* The dataset.head(10) function displays the first 10 rows of the dataset. This initial look helps to understand the structure of the data, the features (columns), and the first few values in each feature.

###### Understanding Features and Target:
* The dataset contains various car attributes (features), such as wheelbase, carlength, curbweight, enginesize, horsepower, etc., along with price, which seems to be the target variable we might want to predict.
* Identifying the target variable is an important part of preprocessing, as it guides feature selection and further steps.

<img src="https://github.com/user-attachments/assets/cab4ef5e-9aae-4acd-a561-8b3f76168681" alt="Screenshot" width="500" height="250">

##### Getting input and output
It is the process for selecting the input (independent variables) and output (dependent variable) from the dataset.

###### This is the step-by-step explanation for getting the inputs and output. (as shown in the image below)

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
* 
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
Similarly:

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
<p align="left">
  <img src="https://github.com/user-attachments/assets/485a0190-f054-4e8e-a6c2-571299437bbe" width="700" height="300" alt="Building and training model-VEED">
</p>

#### MODEL EVALUATION: 
To evaluate the performance of both linear and logistic regression models, appropriate metrics must be used. For linear regression, key metrics include R-squared and Adjusted R-squared and metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), which measure the model’s accuracy in predicting continuous values. (dagdagan ng about sa logistic). These metrics provide a clear view of model reliability, enhancing predictions in both continuous and categorical contexts.

##### For Linear Regression 
<p align="left">
  <img src="https://github.com/user-attachments/assets/37edc2f0-783c-4e3c-908d-39b517a23311" width="700" height="300" alt="Evaluating the model-VEED">
</p>

#### REAL-WORLD APPLICATION: 
The knowledge gained from building and evaluating regression models will be applied to real-world scenarios. Linear regression can be used to predict numerical outcomes such as stock prices, real estate values, and energy consumption. Logistic regression will assist in solving classification problems such as predicting customer churn, detecting fraud, or diagnosing medical conditions based on patient data and so on. By interpreting the outcomes of these models, organizations can make informed decisions, optimize operations, and better understand the factors influencing the predictions.


