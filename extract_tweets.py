#!/usr/bin/env python
# encoding: utf-8

import tweepy 
import csv
from tabulate import tabulate
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import fastai
from fastai import *
from fastai.text import * 

#Twitter API credentials
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""

count = 0
final_sc = []

def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200,tweet_mode='extened')
	
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print ("getting tweets before %s" % (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest,tweet_mode='extened')
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		
		print ("...%s tweets downloaded so far" % (len(alltweets)))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	def getText(data):

    # Try for extended text of original tweet, if RT'd (streamer)
	    try: text = data.retweeted_status.extended_tweet.full_text
	    except: 
	        # Try for extended text of an original tweet, if RT'd (REST API)
	        try: text = data.retweeted_status.full_text
	        except:
	            # Try for extended text of an original tweet (streamer)
	            try: text = data.extended_tweet.full_text
	            except:
	                # Try for extended text of an original tweet (REST API)
	                try: text = data.full_text
	                except:
	                    # Try for basic text of original tweet if RT'd 
	                    try: text = data.retweeted_status.text
	                    except:
	                        # Try for basic text of an original tweet
	                        try: text = data.text
	                        except: 
	                            # Nothing left to check for
	                            text = ''
	    return text
	
	# import IPython
	# IPython.embed()
	
	outtweets = [[tweet.id_str, tweet.created_at, getText(tweet)] for tweet in alltweets]
	df = pd.DataFrame(outtweets)
	
	print(tabulate(df.head(), headers='keys', tablefmt='psql'))
	

	data_clas = load_data('./')
	learn_model = text_classifier_learner(data_clas,arch= AWD_LSTM, drop_mult=0.3)
	learn_model.load("model-2")
	global score 
	score = []
	count = 0
	
	def label(lab):
		x = learn_model.predict(lab)
		score.append(x[2])
		return x[0]

	def change_labels(lab):
		if lab == 1:
			return 'racism'
		else:
			return 'sexism'
	def labelling(lab):
		global count
		global final_sc
		x_val = score[count]
		count+=1
		sor = sorted(x_val, reverse = True)
		first, second = sor[0], sor[1]
		if not(max(x_val) > 0.5 and  (first-second)>=0.3):
			return 0
		
		var = lab.data
		
		if var == 0:
			return 0
		elif var == 2:
			final_sc.append(x_val)
			return 2
		elif var == 1:
			final_sc.append(x_val)
			return 1


	df[3] = df[2].apply(lambda x: label(x))
	df[3] = df[3].apply(lambda x: labelling(x))
	
	
	df_new = df[df[3]!=0]
	df_new[3] = df_new[3].apply(lambda x: change_labels(x))
	df_new[4] = final_sc
	
	#df_new.rename( columns={0: " Tweet ID", 1: "Timestamp",2:"Tweet",3:"Label"})
	df_new.columns = ["Tweet ID","Timestamp","Tweet","Label","scores"]
	
	return df_new
	
	#write the csv	
	# with open('%s_tweets.csv' % screen_name, 'wb') as f:
	# 	writer = csv.writer(f)
	# 	writer.writerow(["id","created_at","text"])
	# 	writer.writerows(outtweets)
	
	# pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("purvak_lapsiya")
