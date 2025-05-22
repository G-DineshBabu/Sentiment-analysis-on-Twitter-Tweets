import tweepy
from predict_sentiment import predict_sentiment

# Replace this with your actual Bearer Token from your Twitter Developer account
bearer_token = "AAAAAAAAAAAAAAAAAAAAAB%2BB1wEAAAAAVZx4fv21JFmc7%2FeehfS52o%2Fw%2BSM%3DU1gaQCGMndLUpDgfUYPqmcuVIgblSEOi8gHmF9rT53WAFGJwjG"

# Initialize Tweepy client
client = tweepy.Client(bearer_token=bearer_token)

# Define the search query (change 'iPhone' to any keyword you want)
query = "iPhone -is:retweet lang:en"

try:
    # Fetch recent tweets (max 10 tweets)
    tweets = client.search_recent_tweets(query=query, max_results=10)

    print("\n🔍 Analyzing Sentiments of Recent Tweets:\n")

    if tweets.data:
        for tweet in tweets.data:
            text = tweet.text
            sentiment, score = predict_sentiment(text)
            print(f"Tweet: {text}")
            print(f"Sentiment: {sentiment} (Confidence: {score:.2f})\n")
    else:
        print("No tweets found for the given query.")

except tweepy.TweepyException as e:
    print(f"An error occurred: {e}")
