
# Product Category Classification

This project uses Machine Learning techniques to classify products into categories based on their titles.

## Dataset

The dataset contains product information such as:
- product title
- merchant ID
- category label
- product code
- number of views
- merchant rating
- listing date

The target variable is `category_label`.

## Project Structure

- `data/products.csv` - dataset
- `notebook/product_category_classification.ipynb` - full analysis and model training
- `src/train_model.py` - script for training and comparing models

## Models Used

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## Workflow

1. Load and explore the dataset
2. Clean column names and handle missing values
3. Preprocess product titles
4. Convert text using TF-IDF
5. Train multiple models
6. Compare model performance
7. Evaluate the best model
8. Test predictions on new data

## Goal

The goal of the project is to predict the correct product category using only the product title.

## Conclusion

The results show that Machine Learning models can effectively classify products based on text data, achieving good accuracy using TF-IDF and classification algorithms.
