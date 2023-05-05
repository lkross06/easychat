# EasyChat

### Lucas Ross and Finn Taylor
### Design Project May 5, 2023

This program creates a chatbot that responds to user messages in the terminal. Using a dataset that categorizes
certain phrases by their intent, a decision tree model was constructed to classify collections of words by
intent, and the chatbot took a response from the dataset to respond. The chatbot uses the __Natural Language Processing Toolkit (NLTK)__ library to process individual words.

**Directory Contents**:
- ``chatbot.py`` main file!! runs a chatbot conversation in the terminal
- ``chatbot.ipynb`` same as chatbot.py, but with visualizations of datasets and decision tree models
- ``user.py`` contains UserData class for accessing previous info in chatbot conversation
- ``preprocessor.py`` contains NLTK functions for processing words
- ``data.json`` dataset of user queries and chatbot responses, categorized by intent
- ``setup.py`` YOU MUST RUN THIS BEFORE CHATBOT.PY OR CHATBOT.IPYNB!! it downloads content from NLTK with ``nltk.download()``
