import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import math
from operator import itemgetter

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def getidf(term):
    if term not in df.keys():
        return -1
    return math.log10(N/df[term])


def getweight(name, term):
    try:
        fil = files.index(name)
    except:
        print("Error: invalid filename")
        return -1
    try:
        if normalize_weights[fil][term] >= 0:
            return normalize_weights[fil][term]
        else:
            return 0
    except:
        return 0


def query(qstring):
    # make query lower case
    qstring = qstring.lower()
    # tokenize the query
    qstring = tokenizer.tokenize(qstring)

    # remove the stop words from the query
    qfiltered = []
    for word in qstring:
        if word not in stop_words:
            qfiltered.append((word))

    # run Porter stemmer on query
    qstemmed = []
    for word in qfiltered:
        qstemmed.append(stemmer.stem(word))

    # query term frequency
    qtf = {}
    for word in qstemmed:
        qtf[word] = qstemmed.count(word)

    # query weights
    qw = {}
    for word, wgt in qtf.items():
        qw[word] = 1 + math.log10(wgt)

    # normalize weights
    qnorm = {}
    denominator = 0
    for wgt in qw.values():
        denominator += (wgt ** 2)
    denominator = math.sqrt(denominator)
    for word, wgt in qw.items():
        qnorm[word] = wgt / denominator

    q = [a for a in qnorm.values()]


    # create posting list for each term in query
    posting_list = []
    for word in qnorm.keys():
        temp = []
        for iterator, doc in enumerate(normalize_weights):
            if word in doc.keys():
                temp.append([files[int(iterator)], doc[word] ])
        posting_list.append(temp)

    # sort each posting list in descending order of TF-IDF weight
    sortedPosList = []
    for arr in posting_list:
        sortedPosList.append(sorted(arr, key=itemgetter(1), reverse=True ))

    # top 10 elements
    top10 = []
    for posList in sortedPosList:
        top10.append(posList[:10])

    dox = []
    for term in top10:
        dox.append([i[0] for i in term])

    doc_list = dox[0]
    for doc in doc_list:
        current = doc
        flag = True
        for check in dox:
            if current not in doc:
                flag = False
        if flag == False:
            continue
        else:
            break
    if flag==True:
        print(current)
    d = []
    for term in top10:
        for pair in term:
            if pair[0] == current:
                d.append(pair[1])
                continue
    index = 0
    for idx, name in enumerate(files):
        if name == current:
            index = idx
            break
    
    score = sum(a * b for a, b in zip(d, q))
    ret = (current, score)
    return ret

# Step 1: read files and make them lowercase
P1 = './presidential_debates'
docs = []
files = []
for filename in os.listdir(P1):
    files.append(filename)
    file = open(os.path.join(P1, filename), "r", encoding='UTF-8')
    temp = file.read()
    file.close() 
    temp = temp.lower()
    docs.append(temp)

N = len(os.listdir(P1)) # total number of documents, used to calculate inverse document frequency

# Step 2: tokenize

tokens = [tokenizer.tokenize(doc) for doc in docs]

# Step 3: remove stop words; Stopwords list from NLTK Corpus
filtered_tokens = []

for doc in tokens:
    temp = []
    for word in doc:
        if word not in stop_words:
            temp.append(word)
    filtered_tokens.append(temp)

# Step 4: Porter stemming
stemmed_tokens = []
for doc in filtered_tokens:
    temp = []
    for word in doc:
        temp.append(stemmer.stem(word))
    stemmed_tokens.append(temp)

# Step 5: computer TF-IDF vector for tokens in each document

# Find all unique terms
unique = []
for doc in stemmed_tokens:
    for word in doc:
        if word not in unique:
            unique.append(word)

# finding document frequency
df = {}
for word in unique:
    temp = 0
    for doc in stemmed_tokens:
        if word in doc:
            temp += 1
    df[word] = temp

# finding term frequency
tf = []
for document in stemmed_tokens:
    temp = {}
    for word in document:
        temp[word] = document.count(word)
    tf.append(temp)

# calculating tf-idf weight for every term in each doc
w = []
for doc in tf:
    temp = {}
    for word, freq in doc.items():
        weight = (1 + math.log10(freq)) * (math.log10(N / df[word]))
        temp[word] = weight
    w.append(temp)

normalize_weights = []
for doc in w:
    denominator = 0
    temp = {}
    for word, val in doc.items():
        denominator += val**2
    denominator=math.sqrt(denominator)
    for word, val in doc.items():
        temp[word] = val/denominator
    normalize_weights.append(temp)

print("(%s, %.12f)" % query("terror attack"))
print("(%s, %.12f)" % query("health insurance wall street"))
