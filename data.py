import pandas as pd
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from textblob import TextBlob


def sentiment(data):

    qf=pd.read_csv("C:\\Users\\Rishabh\\Downloads\\drug\\drugsComTrain_raw.csv")
    pf=pd.read_csv("C:\\Users\\Rishabh\\Downloads\\drug\\drugsComTest_raw.csv")

    df=pd.concat([qf,pf])

    df=df.drop(['uniqueID','drugName','condition','date','usefulCount'],axis='columns')

    def prepro(df):
        tokens=word_tokenize(df)
        stemmer=SnowballStemmer('english',ignore_stopwords=True)
        stemmed=[stemmer.stem(words) for words in tokens]
        words=[word for word in stemmed if word.isalpha()]
        stop_words=set(stopwords.words('english'))
        ss=[w for w in words if not w in stop_words]
        return ss

    df['review']=df['review'].apply(lambda x:prepro(x))

    def sentiment(df):
        df.loc[df.rating>=4,'sentiment']='neutral'
        df.loc[df.rating<4,'sentiment']='negative'
        df.loc[df.rating>=7,'sentiment']='positive'
        return df


    df=sentiment(df)


    y=df['sentiment']
    x=df.drop(['sentiment'],axis='columns')

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer=TfidfVectorizer()
    X_train=vectorizer.fit_transform(x_train.review.astype(str))
    x_test=vectorizer.transform(x_test.review.astype(str))

    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()
    model.fit(X_train,y_train)

    model.score(x_test,y_test)

    x_test=x_test.reshape((-1,1))
    return model.predict(x_test)


