import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


nltk.download("punkt") #used for tokenization
nltk.download('stopwords') # used for removal of stop words ex. is an and the 
nltk.download('wordnet') # Lexical database of english language 
nltk.download('averaged_perceptron_tagger') #pos tagger for assigning a grammatical category to each word in a sentence 

text="Tokenization is the first step in text analytics.The process of breaking down text paragraphs into smaller chunks such as words or sentence is called tokenization."


choice=1

while(choice!=9):
    print("1.Sentence Tokenization")
    print("2.Word Tokenization")
    print("3.Print Stop Words")
    print("4.Remove non alphabetic chars from sentence")
    print("5.Stemming")
    print("6.Lemmatization")
    print("7.POS tagging")
    print("8.Remove stop words from text")
    print("9.Exit")
    choice=int(input("Enter your choice"))
    
    if(choice==1):
        tokenized_text=sent_tokenize(text)
        print(tokenized_text)
    
    if(choice==2):
        tokenized_word=word_tokenize(text)
        print(tokenized_word)
        
    if(choice==3):
        stop_words=set(stopwords.words("english"))
        print(stop_words)
    
    if(choice==4):
        text2="How to remove the non alphabetic 4charter 1244"
        text2=re.sub('[^a-zA-Z]',' ',text2)
        print(text2)
    
    if(choice==5):
        e_words=["wait","waiting","waited","waits","Learning"]
        ps=PorterStemmer()
        for w in e_words:
            print(ps.stem(w))
    
    if(choice==6):
        wordnet_lemmatizer=WordNetLemmatizer()
        text3="Studies Studying cries cry"
        for w in word_tokenize(text3):
            print("Lemma for '{}' is '{}'".format(w,wordnet_lemmatizer.lemmatize(w)))
    
    if(choice==7):
        data = "The pink sweater fit her perfectly"
        words = word_tokenize(data)
        for word in words:
            print(nltk.pos_tag([word]))
    
    if(choice==8):
        tokens=word_tokenize(text.lower())
        filtered_words=[]
        for t in tokens:
            if(t not in stop_words):
                filtered_words.append(t)
        
        print("Tokens : ",tokens)
        print("Filtered tokens : ",filtered_words)


#how to run file
#first run the file - #python3 ass7.py
#if error or library required - 

# 1. Install required libraries (only once)
#pip install nltk pandas

# 2. Run the file
#python3 ass7.py