#!/usr/local/pkg/python/bin/python
__author__ = "Raymond Ordona"
__credits__ = ["Amitesh Shukla", "Haibin Huang"]
__copyright__ = "Copyright 2017, UIUC MCSDS CS410 Project"
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Raymond Ordona"
__email__ = "rmordona@gmail.com"
__status__ = "Open Source Release"

from scipy import sparse
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords #nltk.download('stopwords')
from nltk import pos_tag # nltk.download('averaged_perceptron_tagger') 
from nltk.corpus import sentiwordnet as swn #nltk.download('sentiwordnet')
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, BigramCollocationFinder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer

"""
If you use the VADER sentiment analysis tools, please cite:

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""

#from nltk.metrics import precision,recall

import os
import subprocess
import sys
import getopt
import logging
import pandas as pd
import numpy as np
import math
import hashlib
import toml
#import collections

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.width', 1000)

basename = os.path.basename(__file__)


config_file='./senti_mend.conf'
if os.path.isfile(config_file):
    with open(config_file) as f: 
         toml_content = toml.load(f)
else:
    print ""
    print "Unable to find config file:", config_file
    print ""
    exit(1)

MAX_FEATURES=toml_content['tfidf']['max_features']
nouns = ['NN','NNP','NNS','NNP','NNPS' ]
adjs = ['JJ','JJR','JJS']
verbs = ['VB','VBP','VBD','VBG','VBN', 'VBZ', 'IN']
adverbs = [ 'RB','RBR', 'RBS' ]

book_dataset = toml_content['dataset']['book']
rating_dataset = toml_content['dataset']['rating']
mask_rating_dataset = toml_content['dataset']['mask_rating']
lemmatize_first = toml_content['lemma_pos']['lemmatize_first'] == "True"
sentiment_algo = toml_content['sentiment']['algo']


book_df = pd.read_csv(book_dataset, delimiter="~", 
                         names = [ "Title", "Author", "Category","Description"], encoding="utf-8-sig")

rating_df = pd.read_csv(rating_dataset, delimiter="~", 
                         names = [ "Title", "User", "Rating", "Review", "Published", "Sentiment" ], encoding="utf-8-sig")

books = book_df.Title

reviews = rating_df.Review
reviews_df = pd.DataFrame(reviews, columns=['Review'])

uniq_user = rating_df.User.unique()
uniq_title = rating_df.Title.unique()

n_user = uniq_user.shape[0]
n_title = uniq_title.shape[0]

def IsNan(num):
   isnot = None
   try: 
      isnot = int(num)  
   except: 
      pass
   return isnot == None

def get_title(title):

    mytitle = ""
    if IsNan(title) == True:
      mytitle = title
    else:
      try:
        title_row = book_df.loc[int(title)]
        mytitle = title_row['Title']
      except:
         print ""
         print "Title Id ({}) possibly out or range".format(title)
         print ""
         print "Please use the following command to get list of titles: ", basename, "-l"
         print ""
         exit(1)
    return mytitle

def Hash(user):
    hash = hashlib.sha1(user).hexdigest()
    return hash[:10]+hash[-4:]

def transpose(pos_tag):
    word, tag = pos_tag
    wordings = word.split()
    if wordings[0] == 'not' and tag in 'JJ':
        word = "un" + wordings[1]
    elif wordings[0] == 'not' and tag in ['NN','JJS']:
        word = "dis" + wordings[1]   
    return (word, tag )
        
def lemmatize(tokens_pos):
    lema = []
    for word,tag in tokens_pos:
        _tag = 'n'
        if tag in verbs:
            _tag = 'v'
        elif tag in nouns:
            _tag = 'n'
        elif tag in adjs:
            _tag = 'a'
        elif tag in adverbs:
            _tag = 'r'

        lema.append(lematizer.lemmatize(word,_tag))
    return lema
    
def sentiment_score(pos_senti, neg_senti, term):
    if term.obj_score() < 1: 
        if term.pos_score() > 0:
            pos_senti.append(term.pos_score())
        if term.neg_score() > 0:
            neg_senti.append(term.neg_score())
            
def sentiment_weight(tokens_pos):
    pos_senti = []
    neg_senti = []
    
    for token, pos in tokens_pos:
        logging.debug("Senti: {} {}".format( token, pos))
        if pos in nouns and len(token) > 1:
          for n in [7,6,5,4,3,2,1]:
            try:
                senti_term = swn.senti_synset(token + '.n.' + str(n))
                sentiment_score(pos_senti, neg_senti, senti_term)
                logging.debug("Sentiment Noun 1 {} {} {} {}".format( senti_term, senti_term.pos_score(), senti_term.neg_score(), senti_term.obj_score()))
                #break
            except:
                pass
        
        if pos in adjs and len(token) > 1:
          for n in [7,6,5,4,3,2,1]:
            try:
                senti_term = swn.senti_synset(token + '.a.' + str(n))
                sentiment_score(pos_senti, neg_senti, senti_term)
                logging.debug("Sentiment Adjectives 1 {} {} {} {}".format( senti_term, senti_term.pos_score(), senti_term.neg_score(), senti_term.obj_score()))
                #break
            except:
                pass

        if pos in verbs and len(token) > 1:
          for n in [7,6,5,4,3,2,1]:
            try:
                senti_term = swn.senti_synset(stemmer.stem(token) + '.v.' + str(n))
                sentiment_score(pos_senti, neg_senti, senti_term)
                logging.debug("Sentiment Verbs 1 {} {} {} {}".format( senti_term, senti_term.pos_score(), senti_term.neg_score(), senti_term.obj_score()))
                #break
            except:
                pass

        if pos in adverbs and len(token) > 1:
          for n in [7,6,5,4,3,2,1]:
            try:
                senti_term = swn.senti_synset(stemmer.stem(token) + '.r.' + str(n))
                sentiment_score(pos_senti, neg_senti, senti_term)
                logging.debug("Sentiment Adverbs 1 {} {} {} {}".format( senti_term, senti_term.pos_score(), senti_term.neg_score(), senti_term.obj_score()))
                #break
            except:
                pass
            
    positive = 0
    negative = 0
    pos_sum = np.sum(pos_senti)
    neg_sum = np.sum(neg_senti) 
    if  np.isnan(pos_sum) == False and np.isnan(neg_sum) == False and pos_sum + neg_sum > 0:
        positive = pos_sum / (pos_sum + neg_sum )
        negative = neg_sum / (pos_sum + neg_sum )

    logging.debug("Weights:  Positive Feedbacks: {} Negative Feedbacks: {} Positive Weight: {} Negative Weight: {}".format(  pos_senti, neg_senti, positive, negative ))
    return positive
    
def sentiment_swn(doc):
    
    operators = set(['not','down'])
    stopwords = set(ENGLISH_STOP_WORDS) - operators
    stopwords = stopwords.union(['gonna', 'does','the','of','and','to','in','a','is','that','for','it'])

    # This uses TF-IDF with both unigram and bigram and with maximum words (features) of 3000
    # It uses IDF and stsop words, also discarding numbers (token_pattern)
    # Sublinear set to true to further penalize long documents, which in our case may not be required ( no use in our case )
    #
    tf = TfidfVectorizer( analyzer='word', ngram_range=(1,2), lowercase= True,   min_df=1 , max_df=2, 
                     max_features=MAX_FEATURES, norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=True, 
                     stop_words = stopwords,   token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')

    weight = 0;

    try:

       # apply TF-IDF
       tfidf_matrix =  tf.fit_transform(doc)

       # get pre-processed terms
       feature_names = tf.get_feature_names() 

       if lemmatize_first == True:

          # lemmatize first before pos-tag
          lema_stem =   [lematizer.lemmatize(w)  for w in feature_names]
   
          # apply pos-tags after lemmatize
          tokens_pos = pos_tag(feature_names)

          # now transpost
          tokens_pos = [transpose(term) for term in tokens_pos]

       else:

          # apply pos-tags
          tokens_pos =   pos_tag(feature_names)

          # lemmatize using pos-tags
          tokens_pos = pos_tag( lemmatize(tokens_pos) )

          # now pos-tag it again and transpose
          tokens_pos = [transpose(term) for term in tokens_pos]
  
       logging.debug("POS-Tag {}".format(tokens_pos))

       # finally, get the sentiment weight
       weight =  sentiment_weight( tokens_pos )

    except:
      logging.debug("Error in sentiment ...")
    return weight

def sentiment_vader(doc):

    senti_intense = SentimentIntensityAnalyzer()
    senti_score = senti_intense.polarity_scores(doc[0])
    logging.debug("Senti Vader: {}".format(senti_score))
    pos = senti_score['pos']
    neg = senti_score['neg']
    if (pos + neg ) == 0:
         return 2.5  # neutral
    else:
         return  pos / ( pos + neg )

def sentimentize(doc):
    if sentiment_algo == "swn":
       return sentiment_swn( doc )
    elif sentiment_algo == "vader":
       return sentiment_vader( doc )


def normalize_to_rating(value, max):
    return np.round( value * max )

def normalize_to_1(value,min,max):
    return (value - min ) / ( max - min )

def average(rate, senti):
    if np.isnan(senti) == True or senti == 0:
        return rate
    else:
        senti_rate = normalize_to_rating(senti, 5)
        return np.average([rate, senti_rate])

def rank_titles(other_titles, tit_indexes):
    rank = []
    score = 0.5
    for t in tit_indexes:
        pos = 0.0
        neg = 0.0
        for u in other_titles.iloc[t]:
            if u > 0:
                if u >= 3:
                    pos = pos + 1
                else:
                    neg = neg + 1
        if pos + neg > 0:
            score = pos / ( pos + neg)
        rank.append( { "title" : other_titles.index[t], "score": round(score, 2), "positives": pos } )
    #sorted_rank = sorted(rank, key=lambda d: d['score'], reverse=True)
    return rank
        

def recommend_books(title, matrix):

    # get list of reviewers of title
    try:
      reviewers = matrix.T.loc[title]
    except KeyError:
      print "Book has not been rated yet ... No relevant titles to recommend"
      print ""
      print "To rate book: ", basename, "-r <rate between 1 and 5> -t \"<book title|book id>\" -f \"<feedback>\" -u \"<user>\""
      print ""
      exit(0)
    
    # convert to sparse matrix to remove zeroes
    sparse_reviewers = sparse.csr_matrix(reviewers)
    
    # get other reviewers who have seen the same titles
    other_reviewers = matrix.iloc[sparse_reviewers.indices]

    # get related titles, including title input
    related_titles = sparse.csc_matrix(other_reviewers)

    # convert to sparse matrix, transpose so titles become rows
    sparse_related_titles = sparse.csc_matrix(related_titles.todense().T)

    # get row indexes of titles, eliminate duplicates
    column_index = sparse_related_titles.indices
    tit_indexes = dict.fromkeys(column_index).keys()

    # get other titles
    other_titles = matrix.T.iloc[tit_indexes] 

    # readjust index
    tit_indexes = np.arange(len(tit_indexes))
    
    # now start ranking, return array of dictionary
    rank = rank_titles(other_titles, tit_indexes)
    return pd.DataFrame(rank ).sort_values(['score', 'positives', 'title'], ascending=[False, False, True])
    
    # now rank other titles
    #return rank_titles(other_titles, tit_indexes)


def recommend(title):

    title = get_title(title)

    if len(title) > 0:

    	init_data = np.zeros(shape=(n_user,n_title))

    	matrix = pd.DataFrame(init_data , index=uniq_user, columns = uniq_title)

    	for row in  rating_df.itertuples():
       	   logging.debug("------")
           logging.debug("Review {} {} {}".format("[", row.Review.encode('utf-8'),"]"))
           senti_score = sentimentize([row.Review.encode('utf-8')])
           matrix.at[row.User,row.Title] = average(row.Rating, senti_score )
           logging.debug("Result: Rating: {} Senti Score: {} Normalized Senti: {} Final Score: {}".format( row.Rating, senti_score, normalize_to_rating(senti_score, 5), average(row.Rating, senti_score) ) )

    	sparsity=round(1.0-len(reviews_df)/float(n_user*n_title),3)

        print ""
        print "==============================================================="
        print "                      RECOMMENDATION                           "
        print "==============================================================="
        print ""
    	print 'The sparsity level of Book Reviews is ' +  str(sparsity*100) + '%'
        print ""
        print "Title: ", title
        print ""
        print "Note: The following books received positive score and positive feedback from parents"
        print "      who also read the book ({})".format(title)
        print ""
        recommended = recommend_books(title, matrix)
        print recommended.to_string(index=False)

    else:
        logging.warning("No Title specified ...")

def rate(user,title,rate,feedback):

    mytitle = get_title(title)

    titles = (title for title in book_df.Title.unique())

    if mytitle in titles:

       already_rated = rating_df[['Title','User']].isin([mytitle, user ])

       if True in (already_rated['Title'] & already_rated['User']).values:

         print ""
         print ("User ({}) already rated the title ({})...".format(user, mytitle))
         print ""
         exit(1)
       else:

         feed = "{}~{}~{}~{}~Published.~NONE".format(mytitle,user,rate,feedback)

         print ""
         print "User review recorded:"
         print ""
         print "      Parent User: {}".format(user), "(Hashed)"
         print "       Book Title: {}".format(mytitle)
         print "             Rate: {}".format(rate)
         print "         Feedback: {}".format(feedback)
         print ""
         with open(rating_dataset, "a") as myfile:
             myfile.write(feed + "\n")

    else:
      print ""
      print "No title exists in our list ..."
      print ""
      print "Please use the following command to get list of titles: ", basename, "-l"
      print ""
      exit(1)

def list():
     print book_df[['Title','Category']]

def info(title):

    mytitle = get_title(title)

    titles = (title for title in book_df.Title.unique())

    if mytitle in titles:

      book_info = book_df.loc[book_df['Title'] == mytitle ]
      print ""
      print "Book Information:"
      print ""
      print "               Id: {}".format(book_info.index[0])
      print "            Title: {}".format(book_info['Title'].iloc[0])
      print "           Author: {}".format(book_info['Author'].iloc[0])
      print "         Category: {}".format(book_info['Category'].iloc[0])
      print "      Description: {}".format(book_info['Description'].iloc[0])
      print ""

    else:
      print ""
      print "No title exists in our list ..."
      print ""
      print "Please use the following command to get list of titles: ", basename, "-l"
      print ""
      exit(1)

def search(keyword):

    # If keyword provided is an ID, look up for book id instead
    if IsNan(keyword) == False:
       info(keyword)
       exit(0)

    if len(keyword) == 0:
       print ""
       print "Not enough title keyword to search ..."
       print ""
       print "To search: ", basename, "-s -t <book title>"
       print ""
       exit(1)

    print "Book Information:"
    print ""

    FOUND=0
    for row in book_df.itertuples():
        id = row[0]
        title = row.Title.encode('utf-8')
        author = row.Author.encode('utf-8')
        category = row.Category.encode('utf-8')
        try:
          description = row.Description.encode('utf-8')
        except:
          description = row.Description

        if keyword in title:
           FOUND=1
           print "               Id: {}".format(id)
           print "            Title: {}".format(title)
           print "           Author: {}".format(author)
           print "         Category: {}".format(category)
           print "      Description: {}".format(description)
           print ""
    if FOUND == 0:
        print "No title found ..."


def mask_user():

     for row in rating_df.itertuples():
        title = row.Title.encode('utf-8')
        user = Hash(row.User.encode('utf-8'))
        rating = row.Rating
        comment = row.Review.encode('utf-8')
        published = row.Published

 
        if title.startswith('MM') == True:

           feed = "{}~{}~{}~{}~{}".format(title, user, rating, comment, published)

        else:

           feed = "{}~{}~{}~{}~{}".format(title, user, rating, comment, published)

        with open(mask_rating_dataset, "a") as myfile:
             myfile.write(feed + "\n")

def evaluate(comment):
    print ""
    print "Evaluating ..."
    senti_score = sentimentize([comment])
    print ""
    if senti_score > 0.5:
        print "Positive"
    elif senti_score == 0.5:
        print "Neutral"
    else:
        print "Negative"
    print "" 

def evaluate_all():
    print ""
    print "Precision-Recall analysis  ..."
    print ""
    POS_SENTI=0
    NEG_SENTI=0
    POS_SCORE=0
    NEG_SCORE=0
    TT=0
    FT=0
    TF=0
    FF=0
    for row in rating_df.itertuples():
        title = row.Title.encode('utf-8')
        sentiment = row.Sentiment.encode('utf-8')
        senti_score = sentimentize([row.Review.encode('utf-8')])

        if sentiment == 'POS' and senti_score >= 0.5: TT=TT+1
        if sentiment == 'POS' and senti_score < 0.5: TF=TF+1
        if sentiment == 'NEG' and senti_score >= 0.5: FT=FT+1
        if sentiment == 'NEG' and senti_score < 0.5: FF=FF+1

        #if sentiment == 'POS': POS_SENTI = POS_SENTI + 1
        #if sentiment == 'NEG': NEG_SENTI = NEG_SENTI + 1
        #if senti_score >= 0.5: POS_SCORE = POS_SCORE + 1
        #if senti_score < 0.5: NEG_SCORE = NEG_SCORE + 1
    print "                 S E N T I M E N T  "
    print ""
    print "               POS          NEG " 
    print "S        ----------------------------------"
    print "C   POS        {}    |     {}".format(str(TT).rjust(3), str(FT).rjust(3))
    print "O        ----------------------------------"
    print "R   NEG        {}    |     {}".format(str(TF).rjust(3), str(FF).rjust(3))
    print "E        ----------------------------------"
    print ""
    P = TT / float( TT + TF )
    R = TT / float( TT + FT )
    F1 = ( 2 * P * R ) / float( P + R )
    print "Precision  : ", round(P,2)
    print "Recall     : ",  round(R,2)
    print "F1-MEASURE : ", round(F1,2)
    print ""

def debug():
    logging.basicConfig(level=logging.DEBUG)

def usage():
    print "Usage:"
    print ""
    print "	To list books (simulating listing book):"
    print ""
    print "		", basename, "-l"
    print ""
    print "		", "Note: You can derive the <book id> of a book by running ", basename, "-l"
    print ""
    print "	To display book information:"
    print ""
    print "		", basename, "-i -t <book title|book id>"
    print ""
    print "	To search a book:"
    print ""
    print "		", basename, "-s -t <book title>"
    print ""
    print "	To check for recommended books based on given title:"
    print ""
    print "		", basename, "-c -t <book title|book id> [-d]"
    print ""
    print "		", "where [-d] is in debug mode"
    print ""
    print "	To rate a book (simulating click-throughs and feedback):"
    print ""
    print "		", basename, "-r <rate between 1 and 5> -t \"<book title|book id>\" -f \"<feedback>\" -u \"<user>\""
    print ""
    print "	To evaluate sentiment analysis accuracy:"
    print ""
    print "		", basename, "-e"
    print ""

def main(argv):

    RECOMMEND = 0
    LIST = 0
    DEBUG = 0
    RATE = 0
    INFO=0
    MASK=0
    SEARCH=0
    EVALUATE=0
    TITLE=""
    FEEDBACK=""
    USER=""
    COMMENT=""

    options, remainder = getopt.getopt(argv, 'r:f:t:u:e:dlhcimsa',
	 ['rate=', 'feedback=','title=', 'user=', 'search', 'check', 'help', 'list', 'debug', 'info', 'mask', 'search', 'evaluate' ])

    for opt, arg in options:
        if opt in ('-r', '--rate'):
            RATE = int(arg)
            if RATE > 5 or RATE < 1:
              print RATE
              usage()
              exit(1)
        elif opt in ('-f', '--feedback'):
            FEEDBACK = arg
        elif opt in ('-t', '--title'):
            TITLE = arg
        elif opt in ('-s', '--search'):
            SEARCH = 1
        elif opt in ('-u', '--user'):
            USER = Hash(arg)
        elif opt in ('-c', '--check'):
            RECOMMEND = 1
        elif opt in ('-i', '--info'):
            INFO = 1
        elif opt in ('-l', '--list'):
            LIST = 1
        elif opt in ('-a', '--analyze'):
            EVALUATE=1
        elif opt in ('-e', '--evaluate'):
            COMMENT = arg
        elif opt in ('-m', '--mask'):
            MASK = 1
        elif opt in ('-d', '--debug'):
            DEBUG = 1
        elif opt in ('-h', '--help'):
	    usage()
        else:
            exit(1)

    if DEBUG == 1:
	debug()

    if LIST == 1:
  	list()
    elif INFO == 1:
  	info(TITLE)
    elif SEARCH == 1:
  	search(TITLE)
    elif RECOMMEND == 1:
  	recommend(TITLE)
    elif RATE > 0:
        if len(USER) < 2 or len(FEEDBACK) < 5:
	   usage()
           exit(1)
  	rate(USER,TITLE,RATE,FEEDBACK)
    elif EVALUATE == 1:
  	  evaluate_all()
    elif len(COMMENT) > 0:
          evaluate(COMMENT)
    elif MASK == 1:
        mask_user()


######### Main #############

stemmer = PorterStemmer()
lematizer = WordNetLemmatizer()



if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
