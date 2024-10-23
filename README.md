<p align="center">
  <img src="https://github.com/user-attachments/assets/a9203597-bea5-44ad-beaa-185612b35407" alt="Midterm Banner" width="2500" height="200">
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
#### Linear Regression 
The dataset presents information on understanding the car features that actually influence the price of cars in various countries. In particular, it is the Chinese Automobile Company Geely Auto. They aspire to enter the US market by setting up their manufacturing unit there and producing cars locally. To achieve this, it is essential to identify and understand the factors influencing car pricing in the American market, as these factors may differ significantly from those in the Chinese market. Through the application of linear regression analysis, we can model car prices based on the available independent variables. This approach we can manipulate the design of the cars, the business strategy etc. to meet certain price levels. And this will be a good way for management to understand the pricing dynamics of a new market. 
#### Logistic Regression

### PROJECT OBJECTIVES: 
(WHAT YOU AIM TO ACHIEVE WITH YOUR ANALYSIS)
#### DATA EXPLORATION: 
Analyzing the provided datasets and conducting a thorough analysis of the required data for each scenario involving the application of linear and logistic regression. This includes data organization, cleaning, and the elimination of variables that may introduce obstacles or errors in the final implementation. The process culminates in the integration and application of the refined data to achieve the outcome.

##### For Linear Regression (CAR PRICING ASSIGNMENT)

##### Car Prices Data Dictionary 

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

 <img src="https://github.com/user-attachments/assets/9e010651-7c47-4df2-8f1b-e93f6134bfe0" alt="Screenshot" width="500" height="250">
 
Car_Price Datasheet

<img src="https://github.com/user-attachments/assets/7fb8bcd9-0c7b-4b23-9673-65dfd8f87838" alt="Screenshot" width="500" height="250">

##### Car_Price Datasheet (Cleaned)
-The image below shows the edited datasheet where categorical variables(based on the data dictionary) have been removed for a cleaner and more effective analysis implementation.

<img src="https://github.com/user-attachments/assets/9a8b1f9a-d9c6-42e3-845a-44c7e51e5403" alt="Screenshot" width="500" height="250">

##### Data Preprocessing at VSCode

<img src="https://github.com/user-attachments/assets/cab4ef5e-9aae-4acd-a561-8b3f76168681" alt="Screenshot" width="500" height="250">

<img src="https://github.com/user-attachments/assets/64aea8f5-e456-4e0a-b939-a21e120e7710" alt="Screenshot" width="500" height="250">

<img src="https://github.com/user-attachments/assets/edef7cc1-45de-4299-b846-2a8d3c529285" alt="Screenshot" width="500" height="250">
