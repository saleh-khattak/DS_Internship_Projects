# 🚗 Car Price Prediction using Machine Learning

## 📌 Project Description

This project predicts the **selling price of used cars** using machine
learning techniques. It uses car-related features such as present price,
kilometers driven, fuel type, transmission, seller type, ownership
history, and car age to build a regression model.

This project was developed as part of an **internship task** to
demonstrate an end-to-end machine learning workflow.

------------------------------------------------------------------------

## 🎯 Objectives

-   Collect and preprocess car-related features
-   Perform feature engineering
-   Train a regression model to predict car prices
-   Evaluate the model using standard metrics
-   Understand real-world price prediction applications

------------------------------------------------------------------------

## 🧠 Machine Learning Model

-   **Algorithm:** Random Forest Regressor
-   **Learning Type:** Supervised Learning (Regression)

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Google Colab

------------------------------------------------------------------------

## 📂 Dataset

The dataset contains the following features:

  Feature         Description
  --------------- ----------------------------------
  Selling_Price   Target variable (price in lakhs)
  Present_Price   Current showroom price
  Driven_kms      Distance driven
  Fuel_Type       Petrol / Diesel
  Selling_type    Dealer / Individual
  Transmission    Manual / Automatic
  Owner           Number of previous owners
  Year            Manufacturing year

------------------------------------------------------------------------

## ⚙️ Feature Engineering

-   Converted **Year** into **Car_Age**
-   Removed unnecessary columns like `Car_Name`

------------------------------------------------------------------------

## 📊 Model Evaluation

The model is evaluated using: - Mean Absolute Error (MAE) - Root Mean
Squared Error (RMSE) - R² Score

------------------------------------------------------------------------

## 📈 Visualization

A scatter plot is generated to compare **Actual vs Predicted Prices**.

------------------------------------------------------------------------

## ▶️ How to Run the Project (Google Colab)

1.  Open Google Colab
2.  Upload `car data.csv`
3.  Run the notebook cells
4.  The model will train automatically and display results

------------------------------------------------------------------------

## 🌍 Real-World Applications

-   Used car pricing platforms
-   Automobile resale value estimation
-   Loan and insurance valuation systems
-   Market analysis tools

------------------------------------------------------------------------

## 👤 Author

**Muhammad Saleh**\
Computer Science Student\
Internship Project

------------------------------------------------------------------------

## 📜 License

This project is for educational and internship purposes only.
