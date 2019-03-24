from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import scipy
from sklearn.metrics import precision_score

top_N = 10


def get_queries():
    queries = []
    for i in range(1, 226):
        with open(f'./q/{i}.txt') as f:
            queries.append(f.readline())
    return queries


def get_documents():
    doc = []
    for i in range(1, 1401):
        with open(f'./d/{i}.txt') as f:
            doc.append(f.readline())
    return doc


def get_relevant_docs(id):
    out = []
    with open(f'./r/{id}.txt') as f:
        for doc in f.readlines():
            out.append(int(doc))
        return out


def calculate_euclid(query_vector, data_vectors):
    distances = np.array(
        euclidean_distances(query_vector, data_vectors)[0])

    distances_sorted = distances.argsort() + 1
    return distances_sorted


def calculate_cosine(query_vector, data_vectors):
    distances = np.array(
        cosine_similarity(query_vector, data_vectors)[0])

    distances_sorted = distances.argsort()[::-1] + 1
    return distances_sorted


def process_binary(query):
    documents = get_documents()
    documents.append(query)

    count_vectorizer = CountVectorizer(binary=True)
    count_array = count_vectorizer.fit_transform(documents)

    query_vector = count_array[len(documents) - 1]
    data_vectors = count_array[0:len(documents) - 1]

    euclid = calculate_euclid(query_vector, data_vectors)
    cosine = calculate_cosine(query_vector, data_vectors)

    return euclid, cosine


def process_TF(query):
    documents = get_documents()
    documents.append(query)

    vectorizer = CountVectorizer()
    count_array = vectorizer.fit_transform(documents)

    sums = count_array.sum(1)

    # Number of times term appears in a document) / (Total number of terms in the document
    normalized_array = count_array.multiply(1 / sums)

    normalized_array = scipy.sparse.csr_matrix(normalized_array)

    query_vector = normalized_array[len(documents) - 1]
    data_vectors = normalized_array[0:len(documents) - 1]

    euclid = calculate_euclid(query_vector, data_vectors)
    cosine = calculate_cosine(query_vector, data_vectors)

    return euclid, cosine


def process_TF_IDF(query):
    documents = get_documents()
    documents.append(query)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    query_vector = tfidf_matrix[len(documents) - 1]
    data_vectors = tfidf_matrix[0:len(documents) - 1]

    euclid = calculate_euclid(query_vector, data_vectors)
    cosine = calculate_cosine(query_vector, data_vectors)

    return euclid, cosine


# def compute_precision(id, predicted):
#    reality = get_relevant_docs(id)
#    if len(reality) <= top_N:
#        local_top = len(reality)
#    else:
#        local_top = top_N +1

# print(reality[:local_top])
# print(list(predicted)[:local_top])
#    return precision_score(list(reality[:local_top]), list(predicted)[:local_top],average='micro')

def compute_precision(reality, predicted):
    # P(relevant|retrieved)
    found = 0
    cut_predicted = predicted[:top_N]
    for doc in cut_predicted:
        if doc in reality:
            found += 1
    return found / len(cut_predicted)


def compute_recall(reality, predicted):
    #P(retrieved|relevant)
    found = 0
    cut_predicted = predicted[:top_N]
    for doc in cut_predicted:
        if doc in reality:
            found += 1
    return found / len(reality)


def compute_F_measure(reality, predicted):
    precision = compute_precision(reality, predicted)
    recall = compute_recall(reality, predicted)
    if precision == 0 and recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)



q = get_queries()
f = open('output.csv', 'w')
header = """ID;Binary-Euclid-Precision;Binary-Euclid-Recall;Binary-Euclid-F-measure;Binary-Cosine-Precision;Binary-Cosine-Recall;Binary-Cosine-F-measure;TF-Euclid-Precision;TF-Euclid-Recall;TF-Euclid-F-measure;TF-Cosine-Precision;TF-Cosine-Recall;TF-Cosine-F-measure;TF-IDF-Cosine-Precision;TF-IDF-Euclid-Recall;TF-IDF-Euclid-F-measure;TF-IDF-Cosine-Precision;TF-IDF-Cosine-Recall;TF-IDF-Cosine-F-measure"""
f.write(header + '\n')
for i, query in enumerate(q, 1):
    reality = get_relevant_docs(i)
    q = []

    binary_euclid, binary_cosine = process_binary(query)
    q.append(binary_euclid)
    q.append(binary_cosine)

    TF_euclid, TF_cosine = process_TF(query)
    q.append(TF_euclid)
    q.append(TF_cosine)

    TF_IDF_euclid, TF_IDF_cosine = process_TF_IDF(query)
    q.append(TF_IDF_euclid)
    q.append(TF_IDF_cosine)

    f.write(str(i))
    for job in q:
        f.write(';')
        f.write(str(compute_precision(reality, job)).replace('.',','))
        f.write(';')
        f.write(str(compute_recall(reality, job)).replace('.',','))
        f.write(';')
        f.write(str(compute_F_measure(reality, job)).replace('.',','))

    f.write('\n')

f.close()
