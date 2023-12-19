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

## Feature & Label Assumptions for Reuters-21578 Dataset:

- **Text Features:**
  - Primary sources: The 'Title' and 'Body' attributes collectively contain textual content suitable for classification when processed together.
  - Transformation: Utilizing techniques like TF-IDF or word embeddings facilitates the conversion of text into a numerical format, enhancing model interpretability and performance.

- **Labels:**
  - 'Topics' serve as the primary and relevant labels for prediction tasks, reflecting the diverse categories within the dataset based on binary encoding. These labels, encoded using a binary encoder, provide the necessary ground truth for training and evaluating classification models.

## Objective

1. Construct a multi-class classification model
   
2. Generate a comprehensive classification report to assess the model's performance, incorporating metrics such as accuracy, precision, recall, F1 score, and support
   
## Running the Jupyter Notebook ONLY in Google Colab

Important: The Jupyter Notebook is configured to run seamlessly on Google Colab, allowing for direct downloading of the dataset. It optimally utilizes GPU resources for efficient processing of the BERT Model.

To execute the provided Jupyter Notebook in Google Colab, follow these steps:

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
  - Combine title and body as new label known as text 
  - Perform stopwords removal, stemming, and tokenization.
  - Eliminate special characters, newlines, and numbers.

- **Step 4: Data Exploration / Cleaning**
  - Identify the most frequent words
  - Detect words that occur only once
  - Count the number of unique words

- **Step 5: Prediction using Classifier Model**
  - Vectorize and combine features using both TF-IDF and word embeddings.
  - Convert the data into a binary matrix using MultiLabelBinarizer.
  - Implement multiple classifier models to determine the highest accuracy.
  - Generate a classification report to evaluate and interpret the model's performance

- **Step 6: Prediction based on BERT Transformer**
	- Convert the data to a multi-hot encoding format and establish dictionaries for label mapping.
  - Tokenize the input data and configure the BERT Transformer model for sequence classification.
   - Generate a classification report to evaluate and interpret the model's performance

## Conclusion
The project followed a meticulous approach, encompassing thorough data preprocessing, insightful exploratory data analysis (EDA), and rigorous evaluation of classifier models, including BERT. Through careful handling of features and topics, able to predict the topic when feeding new text. Pickling ensured the dataset's readiness for modeling, although BERT may not be suitable. EDA uncovered essential patterns and relationships within the data, guiding our modeling decisions. The performance evaluation of multiple models using precision, recall, and F1-score metrics provided a comprehensive understanding of their strengths and limitations. The findings contribute valuable insights into the dataset and lay the groundwork for future refinements, emphasizing the continuous improvement of predictive accuracy. Despite BERT performing poorly compared to the Classifier, it may also indicate overfitting in the Classifier.

## Future Works
Exploring Modified Hayes ("ModHayes") Split or cross-validation, as suggested in the dataset readme, would be beneficial. Running more classifier models or fine-tuning BERT could further improve its overall accuracy. It will be great if we can conduct NER analysis to determine category sets for people, places, exchanges and orgs based on title & body. 

## Reference
- https://huggingface.co/datasets/reuters21578
- https://huggingface.co/lxyuan/distilbert-finetuned-reuters21578-multilabel
- https://huda-kassoumeh.medium.com/text-classification-with-reuters-3a4fbdccc60c
- https://github.com/HudaKas/Reuters
- https://github.com/jared-neumann/Reuters-News-Classification/blob/main/Reuters-News-Classification.ipynb
- https://github.com/ersinaksar/NLP-based-Reuters-21578-Automated-News-Classification-with-Naive-Bayes/blob/main/Text%20Classification.ipynb
- https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

