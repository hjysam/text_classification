# Reuters-21578 Text Categorization Collection

## Overview
The Reuters-21578 Text Categorization Collection is an extensive dataset designed for text classification tasks, offering rich attributes for the analysis of news articles. This readme provides an in-depth understanding of the dataset's structure and essential considerations for effective utilization.

## Category Sets
The collection is structured into five distinct category sets:
1. **Topics:** Subjects or themes covered in news articles.
2. **Places:** Geographical locations mentioned in articles.
3. **People:** Names of individuals featured in articles.
4. **Organizations (Orgs):** Names of organizations, companies, or institutions.
5. **Exchanges:** Financial exchanges referenced in articles.

## Feature & Label 
When engaging in text classification on the Reuters-21578 dataset, it is crucial to consider the following:

- **Text Features:**
  - Primary sources: 'Title' and 'Body' attributes contain text content suitable for classification when combined and handled together.
  - Transformation: Techniques such as TF-IDF or word embeddings can convert text into a numerical format.

- **Labels:**
  - 'Topics' serve as common labels for supervised text classification tasks.

## Opening the Jupyter Notebook in Colab

To run the provided Jupyter Notebook in Google Colab, follow these steps:

1. Click on the notebook file (`reuters21578_Text_classification_model.ipynb`) in the project repository.
2. In the top right corner of the GitHub interface, you will see a button labeled "Open in Colab." Click on it.

[Open in Colab](https://colab.research.google.com/github/hjysam/text_classification/blob/main/reuters21578_Text_classification_model.ipynb)

3. The notebook will open in a new tab in Google Colab.

Note: Ensure that you are signed in to your Google account.

## Step Summary 
- **Step 1: Python Module Library**
  - Install the required modules.

- **Step 2: Data Extraction**
  - Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection).
  - Extract data from SGM files to create a dataframe and a data dictionary.
  - Consider using LEWISSPLIT.

- **Step 3: Data Preprocessing**
  - Perform stopwords, stemming, and tokenization.
  - Remove special characters, newlines, and numbers.

- **Step 4: Data Exploration / Cleaning**
  - Identify the most frequent words.
  - Identify words that appear only once.
  - Count the number of unique words.

- **Step 5: Prediction based on Classifier Model**
  - Vectorize and combine using both TF-IDF and word embeddings.
  - Convert into a binary matrix using MultiLabelBinarizer.
  - Run multiple classifier models to determine the best accuracy.
  - Provide a classification report.

- **Step 6: Prediction based on BERT Transformer**
  - Transform to a multi-hot encoding format and create dictionaries for label mapping.
  - Tokenize input data and configure the BERT Transformer model for sequence classification.
  - Provide a classification report.

## Conclusion
The project followed a meticulous approach, encompassing thorough data preprocessing, insightful exploratory data analysis (EDA), and rigorous evaluation of classifier models, including BERT. Through careful handling of features and topics, able to predict the topic when feeding new text. Pickling ensured the dataset's readiness for modeling, although BERT may not be suitable. EDA uncovered essential patterns and relationships within the data, guiding our modeling decisions. The performance evaluation of multiple models using precision, recall, and F1-score metrics provided a comprehensive understanding of their strengths and limitations. The findings contribute valuable insights into the dataset and lay the groundwork for future refinements, emphasizing the continuous improvement of predictive accuracy. Despite BERT performing poorly compared to the Classifier, it may also indicate overfitting in the Classifier.

## Future Works
Exploring Modified Hayes ("ModHayes") Split or cross-validation, as suggested in the dataset readme, would be beneficial. Running more classifier models or fine-tuning BERT could further improve accuracy.

## Reference
- https://huggingface.co/datasets/reuters21578
- https://huggingface.co/lxyuan/distilbert-finetuned-reuters21578-multilabel
- https://huda-kassoumeh.medium.com/text-classification-with-reuters-3a4fbdccc60c
- https://github.com/HudaKas/Reuters
- https://github.com/jared-neumann/Reuters-News-Classification/blob/main/Reuters-News-Classification.ipynb
- https://github.com/ersinaksar/NLP-based-Reuters-21578-Automated-News-Classification-with-Naive-Bayes/blob/main/Text%20Classification.ipynb
-https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

