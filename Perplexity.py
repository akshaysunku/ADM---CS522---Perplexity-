#Copyright 2019, Dan Baciu, Akshay Sunku, Anudeep Paturi, Krishnachaitanya Kilari, Syed Jagirdar, All rights reserved.



import gzip
import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix
import glob
import pandas as pd
import math
from itertools import combinations 
import matplotlib.pyplot as plt

# Following lists and dictionaries are used internally in the following functions
row = []
col = []
data = []
word_store = {}
row2 = []
col2 = []
data2 = []
word_store2 = {}  

# Parse through topic_state file to retrieve words, word_index, and topics
def read_topic_word(path):
    no = 0
    with gzip.open(path,'r') as f:      
        for line in f:
            no = no + 1
            if no>=4:
                words=line.split()
                word=words[4].decode("utf-8")
                word_ind = words[3].decode("utf-8")
                col_ind = words[5].decode("utf-8")
                word_store[word] = int(word_ind)
                append(word_ind, col_ind)
                
# Appends to later be used in sparse matrix
def append(x,y):
    row.append(int(x))
    col.append(int(y))
    data.append(int(1))
    
# Creates topic - word matrix
def word_topic():
    #word-topic matrix
    global word_topic_matrix
    word_topic_matrix=csr_matrix((data, (row, col)), dtype=np.int32)
    row.clear()
    col.clear()
    data.clear()
    matrix = word_topic_matrix.todense()    # Create dense matrix out of sparse matrix
    matrix = matrix + 1
    matrix = pd.DataFrame(matrix)
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    matrix = np.exp(matrix)
    return matrix

# Creates topic - word matrix to be used in Arun2010 calculation
def arun_word_topic():
    #word-topic matrix
    global word_topic_matrix
    word_topic_matrix=csr_matrix((data, (row, col)), dtype=np.int64)
    row.clear()
    col.clear()
    data.clear()
    matrix = word_topic_matrix.todense()    #Create dense matrix out of the sparse matrix
    matrix = matrix + 1
    matrix = pd.DataFrame(matrix.transpose())
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    matrix = np.exp(matrix)
    return matrix

# retrieves values for document - term matrix
def read_document_term(path):
    no = 0
    with gzip.open(path,'r') as f:      
        for line in f:
            no = no + 1
            if no>=4:
                words=line.split()
                word=words[4].decode("utf-8")
                word_ind = words[0].decode("utf-8")
                col_ind = words[3].decode("utf-8")
                word_store2[word] = int(word_ind)
                append2(word_ind, col_ind)

# Used to later create sparse matrix
def append2(x,y):
    row2.append(int(x))
    col2.append(int(y))
    data2.append(int(1))  
              
def document_term():
    #word-topic matrix
    global doc_term_matrix
    doc_term_matrix=csr_matrix((data2, (row2, col2)), dtype=np.int64)
    row2.clear()
    col2.clear()
    data2.clear()
    matrix = doc_term_matrix.todense()  #Create dense document - term matrix
    return pd.DataFrame(matrix)

#Used to retreive document - topic matrix from the composition.txt file
def read_document_topic(path):
    document_topic = pd.read_csv(path, sep='\t', header = None)
    return document_topic

# Deveaud2014 Perplexity Measure 
def Deveaud2014(document_topic):  
    jsd_sum = 0.0
    ncols = len(document_topic.columns)
    
    #taking all the combinations of columns 
    index = [i for i in range(ncols)]
    for k in combinations(index,2):
        p = list(document_topic[k[0]])
        q = list(document_topic[k[1]])
        jsd_sum += jsd(p,q)
    JSD = jsd_sum / ((ncols) * (ncols - 1))                   #dividing the sum of jensen shannon divergences by k * (k-1) 
    print('{0} topics: jsd = {1}'.format(ncols, JSD))                 
    
    return JSD, ncols
    
# Helper function for Deveaud2014
def jsd(p, q):
    x = 0.0
    y = 0.0
    for i in range(len(p)):
        x += p[i] * math.log(p[i]/q[i])
        y += q[i] * math.log(q[i]/p[i])
    return 0.5 * x + 0.5 * y

# CaoJuan2009 Perplexity Measure
def CaoJuan2009(document_topic):
    cos_sum = 0.0
    ncols = len(document_topic.columns)
    
    #taking all the combinations of columns
    index = [i for i in range(ncols)]
    for k in combinations(index,2):
        p = list(document_topic[k[0]])
        q = list(document_topic[k[1]])
        cos_sum += cosineSim(p,q)
    CaoJuan = cos_sum / ((ncols) * (ncols - 1))                   #dividing the sum of cosine similarity by k * (k-1) 
    print('{0} topics: cos = {1}'.format(ncols, CaoJuan))
    
    return CaoJuan, ncols

# Helper function for CaoJuan2009
def cosineSim(p, q):
    p = np.array(p)
    q = np.array(q)
    dot = np.dot(p,q)
    normp = np.linalg.norm(p)
    normq = np.linalg.norm(q)
    cos = dot / (normp * normq)
    return cos

# Arun2010 Perplexity Measure
def Arun2010(L, doc_topic, topic_word, alphas):
    normalize = np.linalg.norm(L)
    kls = 0.0

    cm = []
    cm1 = np.linalg.svd(np.array(topic_word), compute_uv = False)
    # Include sorting by alplha value
    cm2 = L.dot(np.array(doc_topic))
    cm2 /= normalize
    for alpha in alphas.index.values:
        cm.append(cm2[alpha])
    kls = kl(cm1,cm)
    print('{0} topics: kls = {1}'.format(cm1.shape[0], kls))
    return(kls,cm1.shape[0])

# Helper function for Arun2010 to find Kullback-Leibler value
def kl(p, q):
    x = 0.0
    y = 0.0
    for i in range(len(q)):
        x += p[i] * math.log(p[i]/q[i])
        y += q[i] * math.log(q[i]/p[i])
    return x + y

# Retreiving alpha values from topic-state and returning in useful format
def get_alphas(path):
    no = 0
    with gzip.open(path,'r') as f:      
        for line in f:
            no = no + 1
            if no <= 2 and no >1:
                line = line.decode("utf-8").split()
                line = line[2:]
                line = [float(x) for x in line]
                return line

# Plot the results 
def plot_results(results):
    lists = sorted(results.items())
    x,y = zip(*lists)
    plt.plot(x, y)
    plt.xlabel("Topics")
    plt.ylabel("Perplexity")
    plt.show()

# Main method which includes Deveaud2014, CaoJuan2009, and Arun2010 Perplexity Measure
def main():
    # composition_files will hold all composition.txt files in a list
    path1 = r'C:\Users\Akshay Sunku\Desktop\composition'
    composition_files = glob.glob(path1 + "/*.txt")
    # topic_state_files will hold all topic_state files in a list
    path2 = r'C:\Users\Akshay Sunku\Desktop\topic-state'
    topic_state_files = glob.glob(path2 + "/*.gz")
    
    # These dictionaries are used to store the results returned from our three metrics
    jsd_results ={}
    cos_results = {}
    arun_results = {}
    
    # Commence Deveaud2014 Perplexity Measure
    print("Deveaud2014 perplexity results")
    for i in range(len(topic_state_files)):
        #document_topic = read_document_topic(composition_files[i])
        #the first two columns were the index and file location, so we delete those two columns
        #del(document_topic[0])
        #del(document_topic[1])
        #document_topic.columns = range(document_topic.shape[1])
        read_topic_word(topic_state_files[i])
        topic_word_matrix = word_topic()
        JSD, ncols = Deveaud2014(topic_word_matrix)
        jsd_results[ncols] = JSD
    plot_results(jsd_results)
    
    # Commence CaoJuan2009 Perplexity Measure
    '''
    print("CaoJuan2009 perplexity results")
    for i in range(len(composition_files)):
        document_topic = read_document_topic(composition_files[i])
        #the first two columns were the index and file location, so we delete those two columns
        del(document_topic[0])
        del(document_topic[1])
        document_topic.columns = range(document_topic.shape[1])
        #read_topic_word(topic_state_files[i])
        #topic_word_matrix = word_topic()
        COS, ncols = CaoJuan2009(document_topic)
        cos_results[ncols] = COS
    plot_results(cos_results)
    '''
    
    # Commence Arun2010 Perplexity Measure
    print("Arun2010 preplexity results")
    for i in range(len(topic_state_files)):
        #for alphas
        alphas = get_alphas(topic_state_files[i])
        alphas = pd.DataFrame(alphas).sort_values(by=[0],ascending=False)
        #for doc-term matrix
        read_document_term(topic_state_files[i])
        document_term_matrix = document_term()
        #for doc_topic matrix
        document_topic = read_document_topic(composition_files[i])
        del(document_topic[0])
        del(document_topic[1])
        #for topic_word matrix
        read_topic_word(topic_state_files[i])
        topic_word_matrix = arun_word_topic()
        #for count of words per document
        Len = np.array(document_term_matrix.sum(axis = 1, skipna = True))
        
        kl,ncols = Arun2010(Len[:8720], document_topic[:8720], topic_word_matrix, alphas)
        
        arun_results[ncols] = kl
    plot_results(arun_results)
        


if __name__ == "__main__":
    main()