# Reuters-21578 Text Classification

## Overview
The Reuters-21578 Text Classification Collection is an extensive dataset for text tasks, providing rich attributes for news article analysis. This readme explains the dataset's structure and key considerations for effective use.

## Categories
The collection is organized into five category sets:
1. **Topics:** Subjects in news articles.
2. **Places:** Geographical locations.
3. **People:** Names of individuals.
4. **Organizations (Orgs):** Names of companies or institutions.
5. **Exchanges:** Financial exchanges.

## Features & Labels (Assumptions):

Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection).

- **Text Features:**
  - Primary sources: 'Title' and 'Body' for classification.
  - Transformation: TF-IDF or embeddings for numerical conversion.

- **Labels:**
  - 'Topics' as primary labels, binary-encoded for model training.

## Objective

1. Build a multi-class classification model.
2. Generate a comprehensive classification report (accuracy, precision, recall, F1 score).
3. Make predictions on news/articles.

## Jupyter Notebook Setup (Google Colab)

1. [Open Notebook in Colab](https://colab.research.google.com/github/hjysam/text_classification/blob/main/reuters21578_Text_classification_model.ipynb)

## Steps Summary

- **Step 1: Python Modules**
  - Install required modules.

- **Step 2: Data Extraction**
  - Extract data from SGM files, consider LEWISSPLIT.

- **Step 3: Data Preprocessing**
  - Combine title and body as 'text.'
  - Apply stopwords removal, stemming, and tokenization.
  - Eliminate special characters, newlines, and numbers.

- **Step 4: Data Exploration / Cleaning**
  - Identify frequent words, detect unique words.

- **Step 5: Prediction based on Classifier Model**
  - Data preparation, Word2Vec embedding.
  - Train classifiers (Logistic Regression, SGD, Multinomial NB).
  - Evaluate and visualize performance.

- **Step 6: Prediction based on BERT Transformer**
  - Custom data handling, label processing.
  - Tokenization and model preparation.
  - Evaluate model, generate classification report.

## Conclusion

The project covers data preparation, exploration, and testing various models, including BERT. Data setup affects model performance, and exploring topics and document length is crucial. Models like BERT may struggle with large datasets or diverse words, requiring substantial computing power. Handling features and topics carefully ensures better predictions.

Even though BERT didn't outperform the Classifier, it raises questions about model flexibility. Improvements include exploring ensemble methods, dimensionality reduction, or newer transformer models. Future work may involve running more classifiers, fine-tuning BERT, or conducting NER analysis for category sets.

## References
- [Hugging Face - Reuters21578](https://huggingface.co/datasets/reuters21578)
- [DistilBERT Model](https://huggingface.co/lxyuan/distilbert-finetuned-reuters21578-multilabel)
- [Medium - Text Classification with Reuters](https://huda-kassoumeh.medium.com/text-classification-with-reuters-3a4fbdccc60c)
- [GitHub - Reuters News Classification](https://github.com/jared-neumann/Reuters-News-Classification/blob/main/Reuters-News-Classification.ipynb)
- [GitHub - NLP Reuters Classification](https://github.com/ersinaksar/NLP-based-Reuters-21578-Automated-News-Classification-with-Naive-Bayes/blob/main/Text%20Classification.ipynb)
- [Towards Data Science - Multi-Class Text Classification](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)