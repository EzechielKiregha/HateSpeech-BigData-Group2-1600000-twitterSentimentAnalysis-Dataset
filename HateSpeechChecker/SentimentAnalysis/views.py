from django.shortcuts import render
# importing necessary libraries
import re
import pickle
import pandas as pd
import numpy as np

# necessary plotting 
import seaborn as sns
import matplotlib.pyplot as plt

# get feedback randomly
import random

# nltk
# import nltk 
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

def preprocess(textData):
    processedText = []
    
    # Lemmatizer and Stemmer
    wordlemm = WordNetLemmatizer()
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1\+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textData:
        tweet = tweet.lower()
        
        # Replace all links with "URL"
        tweet = re.sub(urlPattern, ' URL', tweet)
        # Replace Emojis from the tweet
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        # Replace @USERAME with USER
        tweet = re.sub(userPattern, ' USER', tweet)
        # Replace all non-Alphabets
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        
        tweetWords = ''
        for word in tweet.split():
            # Check if the word is a stopword.
            # if word not in stepWordList:
            if len(word) > 1:
                # Lemmatize it
                word = wordlemm.lemmatize(word)
                tweetWords += (word+' ')
                
        processedText.append(tweetWords)
        
    return processedText

# t = time.time()
# processesedtext = preprocess(text)
# # print the processing
# print("The Text Preprocessing Complete.")
# print(f"Time Taken: {round(time.time()-t)} seconds")



# Load the vectoriser.
file = open('/home/ezechielwill/Desktop/Python/ML/Hate Speech/vectoriser-ngram-(1,2).pickle', 'rb')
vectoriser = pickle.load(file)
file.close()
# Load the LR Model.
file = open('/home/ezechielwill/Desktop/Python/ML/Hate Speech/Sentiment-LR.pickle', 'rb')
LRmodel = pickle.load(file)
file.close()

# def load_models():
#     '''
#     Replace '..path/' by the path of the saved models.
#     '''
#     return vectoriser, LRmodel

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

# if __name__=="__main__":
#     # Loading the models.
#     #vectoriser, LRmodel = load_models()
    
#     # Text to classify should be in a list.
#     text = ["I hate twitter",
#             "Do they really care about me, They hate me",
#             "Dad I love so much, you are my hero",
#             "May the Force be with you.",
#             "Mr. Stark, I don't feel so good"]
    
#     df = predict(vectoriser, LRmodel, text)
#     print(df.head())
# Text to classify should be in a list.
    
text = ["I hate twitter",
        "Do they really care about me, They hate me",
        "Dad I love so much, you are my hero",
        "May the Force be with you.",
        "Mr. Stark, I don't feel so good"]

# Feedback lists
positive_feedback = [
    "Great positive message!",
    "You spread positivity, keep it up!",
    "Your words inspire others.",
    "Fantastic contribution to the conversation!",
    "You bring joy with your positive words.",
    "Keep sharing your positive thoughts!",
    "Your optimism is contagious.",
    "I appreciate your positive mindset.",
    "Your words make a positive impact.",
    "Well done on promoting positivity!",
    "Thanks for brightening my day with your words.",
    "Your kindness shines through your message.",
    "I love the positive energy in your words.",
    "You have a gift for uplifting others.",
    "Your positivity is a breath of fresh air.",
    "I'm grateful for your positive contribution.",
    "Your message radiates positivity.",
    "You have a talent for spreading happiness.",
    "Your optimism is needed in the world.",
    "Your positive words make a difference."
]

negative_feedback = [
    "Please reconsider your words, let's keep it positive.",
    "Hate speech is not acceptable. Encourage respectful dialogue.",
    "Negative words can hurt. Think about the impact of your message.",
    "Consider expressing disagreement in a more constructive way.",
    "Let's aim for a more positive and inclusive conversation.",
    "Your words may be interpreted negatively. Choose them wisely.",
    "Encourage understanding rather than negativity.",
    "Negative speech can escalate conflicts. Choose peace.",
    "Think about the consequences of spreading negativity.",
    "Consider the impact of your words on others.",
    "Try to find common ground and build bridges.",
    "Negative comments can harm relationships. Choose kindness.",
    "Promote constructive dialogue instead of negativity.",
    "Encourage a positive and respectful environment.",
    "Consider the feelings of others when expressing opinions.",
    "Let's strive for harmony and understanding.",
    "Choose words that build up rather than tear down.",
    "Think about how your words contribute to the conversation.",
    "Your message could be more positive. Encourage optimism.",
    "Seek common ground and understanding in your communication."
]

def predictor(request):
    global text
    if request.method == 'POST':
        input = request.POST['texta']
        text.append(input)
        result = predict(vectoriser, LRmodel, text)
        content = result.to_dict()
        text_cont = content['text']
        sent_cont = content['sentiment']
        data_to_render = []

        for key in text_cont.keys():
            data_to_render.append({
                'key': key,
                'text': text_cont[key],
                'sentiment': sent_cont[key]
            })
        
        lastToList = data_to_render[-1]
        print(lastToList['sentiment'] == 'posi')
        if lastToList['sentiment'] == 'Positive':
            lastToList['evaluated'] = True
        else:
            lastToList['evaluated'] = False
        
        # Randomly select feedback
        lastToList['feedback'] = random.choice(positive_feedback) if lastToList['sentiment'] == 'Positive' else random.choice(negative_feedback)
        
        print(lastToList['evaluated'])
        print(lastToList['feedback'])

        return render(request, 'predict.html', {'currentText': lastToList})

    return render(request, 'predict.html')


def viewsPrevious(request):
    if request.method == 'POST':
        result = predict(vectoriser, LRmodel, text)
        content = result.to_dict()
        text_cont = content['text']
        sent_cont = content['sentiment']
        
        previousTextList = []

        for key in text_cont.keys():
            previousTextList.append({
                'key': key,
                'text': text_cont[key],
                'sentiment': sent_cont[key]
            })
        return render(request, 'predict.html', {'previous': previousTextList})
    return render(request, 'predict.html')