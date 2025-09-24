# IMDB_Movie_MIC
This project classifies IMDB movie reviews as positive or negative using Logistic Regression and Random Forest models.
The goal is to compare model performance using multiple metrics and visualize results.

#Dataset Used:-
IMDB Movie Reviews (CSV format).
Label used for positive is 1 and for negative is 0.
The dataset is split into 80% for training and 20% for testing sets.
TF-IDF vectorization was used to convert text into numeric features, limiting the maximum features to 5000 for efficiency.

#Models Used:-
Models used for training are Logistic Regression and Random Forest.
Logistic Regression is a baseline linear model for binary classification.
Random Forest ensembles decision trees for higher generalization.

#Evaluation Metrics:-
It calculated metrics Accuracy, Precision, Recall, and F1-score. Results were also visualized using confusion matrices and bar plots.

#Logistic Regression:-
Accuracy: 88.98% — correctly predicted nearly 89% of reviews.
Precision: 88.37% — predictions of positive reviews were correct 88% of the time.
Recall: 89.78% — correctly identified nearly 90% of actual positive reviews.
F1-score: 89.07% — strong balance between precision and recall.

#Random Forest:-
Accuracy: 85.37% — slightly lower than Logistic Regression.
Precision: 86.46% — correct in predicting positive reviews about 86% of the time.
Recall: 83.88% — captured roughly 84% of all positive reviews.
F1-score: 85.15% — good overall performance, slightly lower than Logistic Regression.

#Comparison and Observation:-
Logistic Regression outperformed Random Forest in all metrics (accuracy, precision, recall, F1-score) on this dataset.
Logistic Regression is slightly better at identifying positive reviews and provides a better balance between precision and recall.
Random Forest is still a strong model but may be slightly less effective for this particular dataset compared to Logistic Regression.
