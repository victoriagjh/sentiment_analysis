# Improving Computational tools for analyzing Health-Related Social Media Data
##### For UCI-ISURF project
### Abstract
Topics related to health, healthcare, and public health are commonly discussed on social media platforms such as Twitter and Facebook. Such data provide an invaluable source of information for researchers and policy makers to better understand the public’s opinions toward important issues such as health policies (e.g., the Affordable Care Act) and controversial health interventions (e.g., HPV vaccination). <br>
**Sentiment analysis** , a technique rooted in natural language processing, is a commonly used tool for deriving insights from social media data, by identifying positive, neutral, or negative emotions from user-generated text. While there have been many sentiment analysis tools available, most of them were developed outside of the healthcare domain, e.g., based on movie reviews. **Consequently, their performance is suboptimal, and varies to a great extent when applied to health data on social media.**<br>
This project aims to develop an integrated tool to help researchers and policy makers evaluate the performance of competing sentiment analysis algorithms to make informed decisions on which algorithm might perform best based on the characteristics of the dataset being analyzed. To achieve this goal, we developed a web-based application using Python and Django framework that integrates four commonly used sentiment analysis tools: Vader, Textblob, Stanford NLP, and the sentiWordNet Dictionary. <br>
Once the user uploads a dataset, the web application will automatically compute the precision, recall, and confusion matrix using each of these sentiment analysis tools and display the results in a visual presentation conducive to easy assessment of their comparative performance. The results can be then downloaded for further analysis. Our web application also provides additional features that many of the existing sentiment analysis tools do not support, such as analyzing the text at the sentence level rather than at the post/tweet level. We believe that our project will make a valuable contribution to improving the utility and appropriateness of use of sentiment analysis tools to better understand user-generated health text on social media.

### Requirements
#### 1. Turn on the django server
  `Python manage.py runserver`
##### If you changed the models.py, you have to do
`Python manage.py makemigrations` <br>
`Python manage.py migrate`

#### 2. Turn on the stanford NLP server for analysis

`cd stanford-corenlp-full-2018-10-05 java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000`
Link : https://stackoverflow.com/questions/32879532/stanford-nlp-for-python

#### 3. Turn on the Celery
`celery -A sentiment_analysis worker --loglevel=info`

#### 4. Turn on the Checker for sending mail and total analysis
`Python manage.py checker`

