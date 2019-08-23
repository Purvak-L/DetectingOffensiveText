# Detecting offensive comments


![alt text](https://github.com/Purvak-L/DetectingOffensiveText/blob/master/images/ML%20Pipeline.png)

## Data Collection

The Hatespeech dataset by [Zeerak Waseem](https://github.com/zeerakw/hatespeech) is used to the train baseline model. It provides a data set of tweets which have been annotated for hate speech. They provide the ID and the annotation in a tab seperated file. To obtain the individual tweets, use the Twitter API of your choice and query for the ID's provided. In our use case, python library [Tweepy](https://github.com/tweepy/tweepy) is used to extract tweets by ID. 

## Feature Transformation and Engineering

Preprocessing Pipeline – 
The Class can performs 9 type of preprocessing on text like 

-	remove_strip_links    
-	strip_mentions_hashtags
-	remove_special_characters
-	remove_non_ascii
-	to_lowercase
-	remove_punctuation
-	replace_numbers
-	remove_stopwords
-	stem_words
-	lemmatize_verbs


For Feature transformation, the model is transformed and tested against following techniques.

* Word CountVectorizer 
* Word TF-IDF 
* N-gram TF-IDF
* Glove Word2Vec 
* Sentiment Analyzer
* Topic Modelling
* Fast.ai FastText
* BERT


## Training Models

### Machine Learning Models

1. Support Vector Machines - Support vector machine was considered because of it's ability to perform well on sparse dataset. SVM showed high precision for neutral label of 94% but this is attributed to more number of "neutral" instances of dataset. Even after balanancing dataset by oversampling and undersampling, and regularizing - the precision of racist and sexist remarks remained low.

2. Naive Bayes - Naive bayes performed well for undersampled dataset but, was sensitive to False Positives. Regularizing by Grid Search on hyperparameters improved performance.

3. Logistic Regression - Logistic regression is attributed to have high accuracy when the size of dataset is more as compared to Naive Bayes. When the training data size is small relative to the number of features, including regularisation such as Lasso and Ridge regression can help reduce overfitting and result in a more generalised model. Regularizing an oversampled dataset improved precision and recall and gave best results.


### Deep Learning Models

4. Recurrent Neural Network - RNN when trained on this dataset gave inaccurate results because less number of instances. Also, increasing number of layers led to vanishing gradient problem.

5. ULMFit - [ULMFit](https://arxiv.org/abs/1801.06146) has been entirely implemented in v1 of the fastai library. ULMFiT’s pre-trained language model was trained on the Wikitext 103 dataset by Stephen Merity. fast.ai provides an API where this pre-trained model (along with some standard datasets for testing) can be conveniently and easily loaded for any target task before fine-tuning. Our datasets for text classification (or any other supervised NLP tasks) is rather small. This makes it very difficult to train deep neural networks, as they would tend to overfit on these small training data and not generalize well in practice. ULMFit would, in principle, perform well because the model would be able to use its knowledge of the semantics of language acquired from the generative pre-training. Ideally, this transfer can be done from any source task S to a target task T. 

6. DNN Classfier + BERT - BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.
The academic paper which describes BERT in detail and provides full results on a number of tasks can be found [here](https://arxiv.org/abs/1810.04805). BERT is considered here to evaluate performance of model when it better understand the context in which biased words/offensive words are used. It was observed before that double negative sentence made it difficult for previous models to understand. While using BERT, it is evident that model better understand the context of a biased word.

7. Multi step classification ULMFit + BERT - In this experiment, ULMFit was used to detect normal sentence against offensive/biased sentences, whereas BERT is used to determine class of offensive sentences (racist vs sexist). The initial results showed alot of promise but it's difficult to deploy such models. Though accuracy improved significatly, this method is not used in our final product.

### Machine Learning Pipeline

machine_learning.py - Machine Learning file walks through loading data, preprocessing data, feature selection, training and optimizing machine learning models and saving models. This file is structured in object oriented fashion and every class can be pulled out to perform specify task. 

Structure -
Following shows how machine_learning.py is structured

- Class LoadData -
  * LoadData

- Class Preprocessing -
  -	remove_strip_links    
  -	strip_mentions_hashtags
  -	remove_special_characters
  -	remove_non_ascii
  -	to_lowercase
  -	remove_punctuation
  -	replace_numbers
  -	remove_stopwords
  -	stem_words
  -	lemmatize_verbs

- Class Feature Selection -
  * Word CountVectorizer 
  * Word TF-IDF 
  * N-gram TF-IDF
  * Glove Word2Vec (TBA, Currently in ipynb)
  * Sentiment Analyzer (TBA, Currently in ipynb)
  * Topic Modelling (TBA, Currently in ipynb)

- Class Machine Learning -
  * Support Vector Machine
  * Logistic Regression
  * Naive Bayes
  * XGBoost (TBA, Currently in ipynb)
  * RNN (TBA, Currently in ipynb)

Note: The Machine Learning Class also does hyperparameter optimization for models.

You can run this pipeline by:

``` python machine_learning.py```

### Deep learning Models ULMFit Pipeline

The ULMFit model relies on fastai's AWD_LSTM architecture (below). After optimization (dropout, oversampling, num of iterations) we selected this model.

```
SequentialRNN(
  (0): MultiBatchEncoder(
    (module): AWD_LSTM(
      (encoder): Embedding(60003, 300, padding_idx=1)
      (encoder_dp): EmbeddingDropout(
        (emb): Embedding(60003, 300, padding_idx=1)
      )
      (rnns): ModuleList(
        (0): WeightDropout(
          (module): LSTM(300, 1150, batch_first=True)
        )
        (1): WeightDropout(
          (module): LSTM(1150, 1150, batch_first=True)
        )
        (2): WeightDropout(
          (module): LSTM(1150, 300, batch_first=True)
        )
      )
      (input_dp): RNNDropout()
      (hidden_dps): ModuleList(
        (0): RNNDropout()
        (1): RNNDropout()
        (2): RNNDropout()
      )
    )
  )
  (1): PoolingLinearClassifier(
    (layers): Sequential(
      (0): BatchNorm1d(900, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Dropout(p=0.4)
      (2): Linear(in_features=900, out_features=50, bias=True)
      (3): ReLU(inplace)
      (4): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): Dropout(p=0.1)
      (6): Linear(in_features=50, out_features=2, bias=True)
    )
  )
)


```

You can train ULMFit model by running ```ULMFit Models.ipynb``` . This notebook will train and save model for you.
It's recommended to run this notebook on colab.

### BERT Pipeline 

The entire training and serving of BERT Based deep learning model can be found ``` BERT_Training_Serving.ipynb ```. This relies on cased version of BERT. BERT here is tested on hatespeech data which classifies offensive language vs not offensive language. This model was developed with idea of using it as first layer of classification between text. Training it on dataset provided with 93.5% accuracy.

![alt_text](https://github.com/Purvak-L/DetectingOffensiveText/blob/master/images/Screen%20Shot%202019-08-19%20at%204.44.46%20PM.png)

## Front-end and Flask Framework

- flaskblog.py - Flaskblog is a project file that holds all the different endpoints of the application. Home, Simple, Login, Register, about are endpoints served by this flask server.

- login and login.html - This endpoint and html file is used to accept user input in "form" type. After user inputs the twitter handle, login endpoint calls extract_tweets.py which extracts tweets and labels tweets. It then render's simple.html

- simple and simple.html - This endpoint is solely responsible for displaying the dataframe in form table.

- forms.py - This python file is responsible for Flask Form using flask_wtf library. It accepts twitter_handle and password.

- models.py and config.py - This files will be used we want to store userdata in database and configure database.

## Extract Tweets and Load Model

extract_tweets.py - Extract tweets as name suggest leverages tweepy API to extract tweets based on tweet_id. Apart from that, this file helps in loading FastAI's ULMFit model and label tweets. The first half of code extracts and second half labels tweets. If we want to change to different model, modify this file.

Note: extract_tweets.py will need consumer keys and access token. It can be generated from [here](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html)

## Evaluation


### Before Hyperparameter optimization

| Feature\Model |	SVM |	Logistic Regression |	Naïve Bayes |
| --- | --- | --- | --- |
|TF-IDF |	82.15% |	80.77% |	76.84%|
|Count Vector|81.54%|	82.46%|79.32%|
|TF-IDF (n gram)|	78.83%|77.01%|77.93%|

### After Hyperparameter optimization

| Feature\Model |	SVM |	Logistic Regression |	Naïve Bayes |
| --- | --- | --- | --- |
|TF-IDF |	84.15% |	87.56% |	79.03%|
|Count Vector|81.54%|	82.46%|79.06%|
|TF-IDF (n gram)|	78.83%|77.01%|78.01%


### Deep Learning Models

| Feature\Model |	AWD_LSTM | DNN Classifier |
| --- | --- | --- |
|ULMFit |	85.27% |	NA |	
|BERT | NA |	93.45%|
