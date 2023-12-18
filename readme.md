# Reuters-21578 Text Categorization Collection

## Overview
The Reuters-21578 Text Categorization Collection is a comprehensive dataset designed for text classification tasks, providing rich attributes for analyzing news articles. This readme provides an overview of the dataset's structure and key considerations for utilization.

## Category Sets
The collection is organized into five distinct category sets:
1. **Topics:** Subjects or themes of news articles.
2. **Places:** Geographical locations mentioned in articles.
3. **People:** Names of individuals featured in articles.
4. **Organizations (Orgs):** Names of organizations, companies, or institutions.
5. **Exchanges:** Financial exchanges mentioned in articles.

## Feature & Label 
When performing text classification on the Reuters-21578 dataset, consider the following at:

- **Text Features:**
  - Primary source: 'Title' and 'Body' attributes contain text content suitable for classification where both are combined and handle together
  - Transformation: Techniques like TF-IDF, or word embeddings convert text into a numerical format.

- **Labels:**
  - 'Topics serves as common labels for supervised text classification tasks.

## Run the Jupyther Notebook



## Step Summary 
- Step 1: Python Module Library
- Step 2: Data Extraction
  - download from https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection
  - extract from SGM files to create dataframe and data dictionary 
  - consider to use LEWISSPLIT
- Step 3: Data Preprossing
  - performs stopwords, stemming and tokenization 
  - also performs remove special character, newlines and number 
- Step 4:Data Exploration / Cleaning  
  - Identify the most frequent words
  - Identify the words that appear for one time
  - Count the number of uique words 
- Step 5: Prediction based on Classifier Model
  - vectorise and combine for both TD-IDF and word-embeding
  -Convert into binary matrix using MultiLabelBinarizer
  - run mulitple classifer model to check for the best accuracy 
  - Classification report 
- Step 6: BERT model  
- Step 7: Text Input  


## Conclusion
Understanding the dataset's structure, feature types, and available splits is crucial for effectively leveraging the Reuters-21578 collection in text classification tasks. Explore the data, experiment with features, and tailor your approach to the specifics of your classification objectives.