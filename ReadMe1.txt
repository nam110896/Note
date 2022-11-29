import random as rd
import re
import math
import string
import sys


class K_cluster:
    def __init__(self, tweets, k, max_iterations ):
        self.tweets = tweets
        self.k = k
        self.max_iterations=max_iterations
        
    def k_means(self):

        centroids = []

        # initialization, assign random tweets as centroids
        count = 0
        hash_map = {}
        while count < k:
            random_tweet_idx = rd.randint(0, len(self.tweets) - 1)
            if random_tweet_idx not in hash_map:
                count += 1
                hash_map[random_tweet_idx] = True
                centroids.append(self.tweets[random_tweet_idx])

        iter_count = 0
        prev_centroids = []

        # run the iterations until not converged or until the max iteration in not reached
        while (isConverge(prev_centroids, centroids)) == False and (iter_count < self.max_iterations):

            # assignment, assign tweets to the closest centroids
            clusters = new_cluster(self.tweets, centroids)

            # to check if k-means converges, keep track of prev_centroids
            prev_centroids = centroids

            # update, update centroid based on clusters formed
            centroids = new_centroids(clusters)
            iter_count = iter_count + 1

        if (iter_count == self.max_iterations):
            print("max iterations reached, K means not converged")
        else:
            print("After ", str(iter_count) +" times running iteration" +", and the resullt is:")

        sse = SSE_function(clusters)

        return clusters, sse, centroids



def new_cluster(tweets, centroids):

    clusters = {}

        # for every tweet iterate each centroid and assign closest centroid to a it
    for t in range(len(tweets)):
        min_dis = math.inf
        cluster_idx = -1
        for c in range(len(centroids)):
            dis = jaccard_Distance(centroids[c], tweets[t])
                # look for a closest centroid for a tweet
            if centroids[c] == tweets[t]:
                # print("tweet and centroid are equal with c: " + str(c) + ", t" + str(t))
                cluster_idx = c
                min_dis = 0
                break

            if dis < min_dis:
                cluster_idx = c
                min_dis = dis

            # randomise the centroid assignment to a tweet if nothing is common
        if min_dis == 1:
            cluster_idx = rd.randint(0, len(centroids) - 1)

            # assign the closest centroid to a tweet
        clusters.setdefault(cluster_idx, []).append([tweets[t]])
            # print("tweet t: " + str(t) + " is assigned to cluster c: " + str(cluster_idx))
            # add the tweet distance from its closest centroid to compute sse in the end
        last_tweet_idx = len(clusters.setdefault(cluster_idx, [])) - 1
        clusters.setdefault(cluster_idx, [])[last_tweet_idx].append(min_dis)

    return clusters


def new_centroids(clusters):

    centroids = []

        # iterate each cluster and check for a tweet with closest distance sum with all other tweets in the same cluster
        # select that tweet as the centroid for the cluster
    for c in range(len(clusters)):
        min_dis_sum = math.inf
        centroid_idx = -1

        # to avoid redundant calculations
        min_dis_dp = []

        for t1 in range(len(clusters[c])):
            min_dis_dp.append([])
            dis_sum = 0
            # get distances sum for every of tweet t1 with every tweet t2 in a same cluster
            for t2 in range(len(clusters[c])):
                if t1 != t2:
                    if t2 < t1:
                        dis = min_dis_dp[t2][t1]
                    else:
                        dis = jaccard_Distance(clusters[c][t1][0], clusters[c][t2][0])

                    min_dis_dp[t1].append(dis)
                    dis_sum += dis
                else:
                    min_dis_dp[t1].append(0)

            # select the tweet with the minimum distances sum as the centroid for the cluster
            if dis_sum < min_dis_sum:
                min_dis_sum = dis_sum
                centroid_idx = t1

        # append the selected tweet to the centroid list
        centroids.append(clusters[c][centroid_idx][0])

    return centroids

def isConverge(prev_centroid, new_centroids):

    # false if lengths are not equal
    if len(prev_centroid) != len(new_centroids):
        return False

        # iterate over each entry of clusters and check if they are same
    for c in range(len(new_centroids)):
        if " ".join(new_centroids[c]) != " ".join(prev_centroid[c]):
            return False

    return True

def SSE_function(clusters):

    sse = 0
    # iterate every cluster 'c', compute SSE as the sum of square of distances of the tweet from it's centroid
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            sse = sse + pow(clusters[c][t][1],2)
    return sse

def jaccard_Distance(tweet1, tweet2):

    # return the jaccard distance
    return 1 - (len(set(tweet1).intersection(tweet2)) / len(set().union(tweet1, tweet2)))

if __name__ == '__main__':

    f = open(str(sys.argv[1]), "r")
    tweets = list(f)
    tweets_data = []

    for i in range(len(tweets)):

        # remove \n from the end after every sentence
        tweets[i] = tweets[i].strip('\n')

        # Remove the tweet id and timestamp
        tweets[i] = tweets[i][50:]

        # Remove any word that starts with the symbol @
        tweets[i] = " ".join(filter(lambda x: x[0] != '@', tweets[i].split()))

        # Remove any URL
        
        tweets[i] = re.sub(r"www\S+", "", tweets[i])
        tweets[i] = re.sub(r"http\S+", "", tweets[i])

        # remove colons from the end of the sentences (if any) after removing url
        tweets[i] = tweets[i].strip()
        tweet_len = len(tweets[i])
        if tweet_len > 0:
            if tweets[i][len(tweets[i]) - 1] == ':':
                tweets[i] = tweets[i][:len(tweets[i]) - 1]
        # Convert every word to lowercase
        tweets[i] = tweets[i].lower()
        # Remove any hash-tags symbols
        tweets[i] = tweets[i].replace('#', '')
        # trim extra spaces
        tweets[i] = " ".join(tweets[i].split())
        # remove punctuations
        tweets[i] = tweets[i].translate(str.maketrans('', '', string.punctuation))

        # convert each tweet from string type to as list<string> using " " as a delimiter
        tweets_data.append(tweets[i].split(' '))

    f.close()

    # default value of K for K-means
    k = int(sys.argv[2])
    kc = K_cluster(tweets_data,k,100)
    # for every experiment 'e', run K-means
    print("Running K _ Cluster Algorithm " +  " with k = " + str(k))

    clusters, sse ,centroids= kc.k_means()

    # for every cluster 'c', print size of each cluster

    for c in range(len(clusters)):
        print(str(c+1) + ": The centroid is " + str(centroids[c])+ " include " + str(len(clusters[c])) + " tweets")
           
    print("--> SSE : " + str(sse)+ '\n')
