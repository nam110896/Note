import sys
import random as rd
import math
import string
import re

class K_cluster:
    def __init__(self,tweets,k,max_iteractions=100):
        self.tweets = tweets
        self.k = k
        self.max_iteractions = max_iteractions
        self.clusters= {}
        self.centroids = []
        self.sse = 0  # Error sum of saquare

    def K_means(self):
        list = {}
        count = 0
        iter_num = 0
        pre_centroids = []

        while count < self.k:
            random_index = rd.randint(0,len(self.tweets)-1)
            if random_index not in list:
                list[random_index] = True
                self.centroids.append(self.tweets[random_index])
                count +=1
        while (isCover(pre_centroids,self.centroids))!= True and (iter_num < self.max_iteractions):
            self.clusters = new_cluster (self.tweets,self.centroids)
            pre_centroids = self.centroids
            self.centroids = new_centroid(self.clusters)
            iter_num += 1
        self.sse = SSE_function(self.clusters)
        if (iter_num != self.max_iteractions):
            print ("\nAfter " + str(iter_num) +" time running iteration, and the result is: \n")
        else:
            print ("\n Reached max iteraction, K mean can not be converged.")
        self.sse = SSE_function(self.clusters)
        return self.centroids,self.clusters,self.sse

def new_cluster(tweets, centroids):
    cluster_list = {}
    for i in range (len(tweets)):
        get_min = math.inf
        cluster_index = -1
        for j in range(len(centroids)):
            distance = jaccard_Distance(centroids[j],tweets[i])
            if (centroids[j]==tweets[i]):
                cluster_index = j
                get_min = 0
                break
            if distance < get_min:
                cluster_index = j
                get_min = distance
        if get_min == 1:
            cluster_index = rd.randint(0,len(centroids)-1)
        cluster_list.setdefault(cluster_index,[]).append([tweets[i]])
        last_index = len(cluster_list.setdefault(cluster_index,[])) -1
        cluster_list.setdefault(cluster_index,[])[last_index].append(get_min)
    return cluster_list
   
def new_centroid(clusters):
    centroid = []
    for i in range (len(clusters)):
        min_sum = math.inf
        centroid_index = -1
        redundant = []

        for j in range (len(clusters[i])):
            sum_distance = 0
            redundant.append([])
            for k in range (len(clusters[i])):
                if j == k:
                    redundant[j].append(0)
                else:
                    if k>=j:
                        distance = distance = jaccard_Distance(clusters[i][j][0],clusters[i][k][0])
                    else:
                        distance = redundant[k][j]
                    redundant[j].append(distance)
                    sum_distance += distance
            if sum_distance < min_sum: 
                min_sum = sum_distance
                centroid_index = j
        centroid.append(clusters[i][centroid_index][0])
    return centroid

def isCover(pre_centroid,new_centroid):
    if (len(pre_centroid)==len(new_centroid)):
        a = " "
        b = " "
        for i in range(len(new_centroid)):
            if str(a.join(pre_centroid[i]))!= str(b.join(new_centroid[i])):
                return False
        return True
    else:
        return False
    
def jaccard_Distance(A,B):
    
    if (len(set().union(A,B))) == 0:
        return 1
    return 1 - (len(set(A).intersection(B)))/(len(set().union(A,B)))
    


def SSE_function(clusters):
    sse = 0
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            sse += pow(clusters[i][j][1],2)
    return sse
    
def run (temp,k):
    tweets_data = []
    f = open(temp,"r",encoding="mac_roman")
    tweets = list(f)

    for i in range(len(tweets)):
        empty_str = " "
        tweets[i] = tweets[i].strip('\n')
        tweets[i] = tweets[i][50:]
        tweets[i] = empty_str.join(filter(lambda x: x[0] != '@', tweets[i].split()))
        tweets[i] = re.sub(r"www\S+","",tweets[i])
        tweets[i] = re.sub(r"http\S+","",tweets[i])
        tweets[i] = tweets[i].strip()
        if (len(tweets[i])>0):
            if tweets[i][len(tweets[i])-1] == ':':
                tweets[i] = tweets[i][:len(tweets[i])-1]
        tweets[i] = tweets[i].upper()
        tweets[i] = tweets[i].replace('#','')
        empty_str = " "
        tweets[i] = empty_str.join(tweets[i].split())
        tweets[i] = tweets[i].translate(str.maketrans('','',string.punctuation))
        tweets_data.append(tweets[i].split(' '))
    f.close()

    KC = K_cluster(tweets,k,100)
    print("Running K_cluster Algorithm with K = "+str(k)+"\n")
    centroid,cluster,sse = KC.K_means()

    for i in range(len(cluster)):
        print(str(i)+ ": The centroid is "+ str(centroid[i])+" include "+str(len(cluster[i]))+ " Tweets")
    print("_____ SSE = "+str(sse)+" _____\n")        
   

if __name__ == '__main__':

    temp = str(sys.argv[1])
    
    k = int(sys.argv[2])
    if(temp != "ALL"):
        print("\n----------Running  Health-Tweets/"+temp+"-------------\n")
        run(temp,k)
    if(temp == "ALL"):
        print("\n----------Running  Health-Tweets/bbchealth.txt-------------\n")
        run("Health-Tweets/bbchealth.txt",k)
        print("\n----------Running  Health-Tweets/cbchealth.txt-------------\n")
        run("Health-Tweets/cbchealth.txt",k)
        print("\n----------Running  Health-Tweets/cnnhealth.txt-------------\n")
        run("Health-Tweets/cnnhealth.txt",k)
        print("\n----------Running  Health-Tweets/everydayhealth.txt-------------\n")
        run("Health-Tweets/everydayhealth.txt",k)
        print("\n----------Running  Health-Tweets/foxnewshealth.txt-------------\n")
        run("Health-Tweets/foxnewshealth.txt",k)
        print("\n----------Running  Health-Tweets/gdnhealthcare.txt-------------\n")
        run("Health-Tweets/gdnhealthcare.txt",k)
        print("\n----------Running  Health-Tweets/goodhealth.txt-------------\n")
        run("Health-Tweets/goodhealth.txt",k)
        print("\n----------Running  Health-Tweets/KaiserHealthNews.txt-------------\n")
        run("Health-Tweets/KaiserHealthNews.txt",k)
        print("\n----------Running  Health-Tweets/latimeshealth.txt-------------\n")
        run("Health-Tweets/latimeshealth.txt",k)
        print("\n----------Running  Health-Tweets/msnhealthnews.txt-------------\n")
        run("Health-Tweets/msnhealthnews.txt",k)
        print("\n----------Running  Health-Tweets/NBChealth.txt-------------\n")
        run("Health-Tweets/NBChealth.txt",k)
        print("\n----------Running  Health-Tweets/gdnhealthcare.txt-------------\n")
        run("Health-Tweets/gdnhealthcare.txt",k)
        print("\n----------Running  Health-Tweets/nytimeshealth.txt-------------\n")
        run("Health-Tweets/nytimeshealth.txt",k)
        print("\n----------Running  Health-Tweets/nprhealth.txt-------------\n")
        run("Health-Tweets/nprhealth.txt",k)
        print("\n----------Running  Health-Tweets/reuters_health.txt-------------\n")
        run("Health-Tweets/reuters_health.txt",k)
        print("\n----------Running  Health-Tweets/reuters_health.txt-------------\n")
        run("Health-Tweets/reuters_health.txt",k)
        print("\n----------Running  Health-Tweets/wsjhealth.txt-------------\n")
        run("Health-Tweets/wsjhealth.txt",k)
        




