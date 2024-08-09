# ClassificationModel_Assg

# Approach
This project(assignment)’s aim was to classify research paper abstracts into the 7 given domains. First, as always, I started off with cleaning the dataset, no matter good the source. I started off with removing duplicate rows, handled missing values, converted all letters to lowercase, removed whitespaces, HTML links, and used NLTK to remove stop words, as well as low content abstract rows. I wanted to test my abilities to make a good model, so I listed down a few models I wanted to work with, which are what took shape and are listed below. I started off with the basics, LR, SVM, NB. This then led me to use ensemble learning, where I used stacking and voting. I followed this up by using bagging and boosting (random forest and gradient boosting), which ended up taking considerably longer, and made me implement XG Boost with encoded label values.

# PATH
1.	Loaded the training and validation data from local files.
2.	Split the data into testing and training splits, as well as their columns (features and labels)
    - Added Encoding for XG Boost training
3.	Vectorized the text using TF-IDF as it helps to capture and retain the importance of words within a document, with respect to each other. Suggested in text classification
4.	Selected a bunch of models (a lot) and trained over them
    - Also used hyperparameter tuning in one file-case
5.	Evaluated the models, one by one

# MODELS USED

```Bag_Boost.py```:
- Random Forest (Bagging)
- Gradient Boosting (Boosting)

```Ensemble_stacking.py```:
- Stacking Classifier (LR, SVM, NB)

```Ensemble_voting.py```:
- Voting Classifier (LR, SVM, NB)

```Main_modular.py``` [incl. hyperparameter tuning]:
- Logistic Regression
- Linear Support Vector Machine (SVM)
- Multinomial Naive Bayes

```XGBoost.py```:
- XG Boost

# Outputs

## Bag_boost.py:

<p align="center">
  <img src="https://raw.githubusercontent.com/PoyBoi/ClassificationModel_Assg/main/images_static/Screenshot%202024-08-09%20153941.png">
  <br>
</p>
 
## Ensemble_stacking.py:

<p align="center">
  <img src="https://raw.githubusercontent.com/PoyBoi/ClassificationModel_Assg/main/images_static/Screenshot%202024-08-09%20154033.png">
  <br>
</p>
 
## Ensemble_voting.py:

<p align="center">
  <img src="https://raw.githubusercontent.com/PoyBoi/ClassificationModel_Assg/main/images_static/Screenshot%202024-08-09%20154109.png">
  <br>
</p>

## XGBoost.py:

<p align="center">
  <img src="https://raw.githubusercontent.com/PoyBoi/ClassificationModel_Assg/main/images_static/Screenshot%202024-08-09%20154256.png">
  <br>
</p>
 
## Main_modular.py:

<p align="center">
  <img src="https://raw.githubusercontent.com/PoyBoi/ClassificationModel_Assg/main/images_static/Screenshot%202024-08-09%20153904.png">
  <br>
</p>
 
# Assumptions

The key assumptions being made here are:
1.	The data given is sufficient and is enough to train a model 
    - We would never know the correct side of this unless we actually get more data and train and see for ourselves on the basis of the results
2.	The data given is correctly labelled 
3.	The abstract given belongs to only one of the categories provided
4.	Imbalance of data skewed in favor of one label or another

# Future Scope
1.	Can use more sophisticated methods which encapsulate word2vec, fastText, as well as using semantic aware models
2.	Train the model on each Domain’s sub-domain to make it achieve mastery
3.	Can implement data augmentation to combat the lack of small dataset
4.	Can try out more methods, especially for the Voting Classifiers, and the Linear Regression Solvers.  