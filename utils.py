import numpy as np
from numpy import linalg as LA
from scipy.stats import pearsonr
import nltk
import io
import torch
import random
import re
import os
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.spatial.distance import cosine
from scipy.linalg import svd
import sys
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
device = torch.device("cuda")

EPS = 5e-7
#################################find closest points to cluster centers###########################
def closest_points(points, cluster_centers, labels):
    closest_ids = []
    for i in range(cluster_centers.shape[0]):
        ids = np.nonzero(labels == i)
        points_in_cluster = points[labels == i, :]
        if points_in_cluster.shape[0] == 0:
            continue
        closest, _ = pairwise_distances_argmin_min(cluster_centers[i:i+1, :], points_in_cluster)
        id_closest = ids[0][closest[0]]
        closest_ids.append(id_closest)
    return sorted(closest_ids)
##################################################################################################
###########################################Start GEM##############################################
def gs(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    A_c = np.copy(A)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A_c[:, k], A_c[:, k]))
        if R[k, k] < EPS:
            R[k, k] = 0
            continue
        Q[:, k] = A_c[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A_c[:, j])
            A_c[:, j] = A_c[:, j] - R[k, j]*Q[:, k]
    return Q, R

def sent_to_tokens(sent, tokenizer):
    tokens = tokenizer.tokenize(sent)
    return tokens

def rm_pr(m, C_0):
    if C_0.ndim == 1:
        C_0 = np.reshape(C_0, [-1, 1])

    w = np.transpose(C_0).dot(m)
    return m - C_0.dot(w)

def ngram(s_num, C_0, sgv_c, win_sz = 7):
    n_pc = np.shape(C_0)[1]
    num_words = np.shape(s_num)[1]
    wgt = np.zeros(num_words)

    for i in range(num_words):
        beg_id = max(i - win_sz, 0)
        end_id = min(i + win_sz, num_words - 1)
        ctx_ids = list(range(beg_id, i)) + list(range(i+1, end_id + 1))
        m_svd = np.concatenate((s_num[:, ctx_ids], (s_num[:, i])[:, np.newaxis]), axis = 1)

        U, sgv, _ = LA.svd(m_svd, full_matrices = False)

        l_win = np.shape(U)[1]
        q, r = gs(m_svd)
        norm = LA.norm(s_num[:, i], 2)

        w = q[:, -1].dot(U)
        w_sum = LA.norm(w*sgv, 2)/l_win

        kk = sgv_c*(q[:, -1].dot(C_0))
        wgt[i] = np.exp(r[-1, -1]/norm) + w_sum + np.exp((-LA.norm(kk, 2))/n_pc)
    # print wgt
    return wgt

def sent_to_ids(sent, tokenizer):
    """
    sent is a string of chars, return a list of word ids
    """
    tokens = sent_to_tokens(sent, tokenizer)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids

def str_2_num(s1):
    input = tokenizer(s1, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        scores, matrix = model(input['input_ids'].to(device), output_hidden_states=True)
    matrix = matrix[0].squeeze().T.cpu().numpy()
    return matrix

def svd_sv(s1, factor = 3):
    s_num = str_2_num(s1)
    U, s, Vh = LA.svd(s_num, full_matrices = False)
    vc = U.dot(s**factor)
    return vc

def feat_extract(m1, n_rm, C_all, soc):
    w1 = LA.norm(np.transpose(m1).dot(C_all)*soc, axis = 0)
    id1 = w1.argsort()[-n_rm:]
    return id1

def encoder(encoding_list, mymodel, mytokenizer, dim = 768, n_rm = 17, max_n = 45, win_sz = 7):
    """
    corpus_list: the list of corpus, in the case of STS benchmark, it's s1 + s2
    encoding_list: the list of sentences to encode
    dim: the dimension of sentence vector
    """
    global model
    model = mymodel
    global tokenizer
    tokenizer = mytokenizer
    global device
    device = torch.device("cuda")
    
    s_univ = np.zeros((dim, len(encoding_list)))
    encoded = []
    for j, sent in enumerate(encoding_list):
        s_univ[:, j] = svd_sv(sent)
    U, s, V = LA.svd(s_univ, full_matrices = False)
    C_all = U[:, :max_n]
    soc = s[:max_n]
    for j, sent in enumerate(encoding_list):
        m = str_2_num(sent)
        id1 = feat_extract(m, n_rm, C_all, soc)
        C_1 = C_all[:, id1]
        sgv = soc[id1]
        m_rm = rm_pr(m, C_1)
        v = m_rm.dot(ngram(m, C_1, sgv, win_sz))
        encoded.append(torch.from_numpy(v))
    return encoded

###################################################################################


#################################Start Usif########################################

class word2prob(object):
    """Map words to their probabilities."""
    def __init__(self, vectorizer, sentences):
        """Initialize a word2prob object.
        Args:
            count_fn: word count file name (one word per line) 
        """
        self.prob = {}
        total = 0.0

        sparse_mtrx = vectorizer.fit_transform(sentences)
        vocab = vectorizer.vocabulary_
        weights = sparse_mtrx.sum(axis=0)
        total = sparse_mtrx.sum()

        self.prob = { token: (weights[0, id] / total) for token, id in vocab.items() }
        self.min_prob = min(self.prob.values())
        self.max_prob = max(self.prob.values())
        self.count = total

    def __getitem__(self, w):
        word = w[0] + w[1:].lower() # because the first letter is 'G'(\u0120) due to tokenizer
        return self.prob.get(word, self.min_prob)

    def __contains__(self, w):
        word = w[0] + w[1:].lower() # because the first letter is 'G'(\u0120) due to tokenizer
        return word in self.prob

    def __len__(self):
        return len(self.prob)

    def vocab(self):
        return iter(self.prob.keys())

    
    
class uSIF(object):
    """Embed sentences using unsupervised smoothed inverse frequency."""
    def __init__(self, model, tokenizer, prob, n=2, m=5):
        """Initialize a sent2vec object.
        Variable names (e.g., alpha, a) all carry over from the paper.
        Args:
            vec: word2vec object
            prob: word2prob object
            n: expected random walk length. This is the avg sentence length, which
            should be estimated from a large representative sample. For STS
            tasks, n ~ 11. n should be a positive integer.
            m: number of common discourse vectors (in practice, no more than 5 needed)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.m = m

        if not (isinstance(n, int) and n > 0):
            raise TypeError("n should be a positive integer")

        vocab_size = float(len(prob))
        threshold = 1 - (1 - 1/vocab_size) ** n
        alpha = len([ w for w in prob.vocab() if prob[w] > threshold ]) / vocab_size
        assert alpha != 0
        Z = 0.5 * vocab_size
        self.a = (1 - alpha)/(alpha * Z)
        self.weight = lambda word: (self.a / (0.5 * self.a + prob[word])) 

    def _to_vec(self, sentence):
        """Vectorize a given sentence.

        Args:
            sentence: a sentence (string) 
        """
        # regex for non-punctuation
        not_punc = re.compile('.*[A-Za-z0-9].*')

        # preprocess a given token
#         def preprocess(t):
#             t = t.lower().strip("';.:()").strip('"')
#             t = 'not' if t == "n't" else t
#             return re.split(r'[-]', t)
        #sentence = sentence.lower()
        tokens = self.tokenizer.tokenize(sentence)
        input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            scores, hiddens = self.model(input['input_ids'].to(device), output_hidden_states=True)
            v_t = hiddens[0].squeeze().cpu().numpy()
#         for token in word_tokenize(sentence):
#             if not_punc.match(token):
#                 tokens = tokens + preprocess(token)

        #tokens = list(filter(lambda t: t in self.vec, tokens))
        
        # if no parseable tokens, return a vector of a's        
#         if tokens == []:
#             return np.zeros(300) + self.a
#         else:
        #v_t = np.array([ self.vec[t] for t in tokens ])
        v_t = v_t * (1.0 / np.linalg.norm(v_t, axis=0))
        v_t = np.array([ self.weight(t) * v_t[i,:] for i,t in enumerate(tokens) ])
        return np.mean(v_t, axis=0)

    def embed(self, sentences):
        """Embed a list of sentences.
        Args:
            sentences: a list of sentences (strings)
        """
        vectors = [ self._to_vec(s) for s in sentences ]

        if self.m == 0:
            return vectors

        proj = lambda a, b: a.dot(b.transpose()) * b
        svd = TruncatedSVD(n_components=self.m, random_state=0).fit(vectors)
  
        # remove the weighted projections on the common discourse vectors
        for i in range(self.m):
            lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
            pc = svd.components_[i]
            vectors = [ v_s - lambda_i * proj(v_s, pc) for v_s in vectors ]

        return np.array(vectors)
 


def get_paranmt_usif(sentences, model, tokenizer, vectorizer):
    """Return a uSIF embedding model that used pre-trained ParaNMT word vectors."""
    prob = word2prob(vectorizer, sentences)
    usif  = uSIF(model, tokenizer, prob)
    return usif.embed(sentences)
#########################################################################################


#####################################Start S3E###########################################
def create_weights(sentences, vectorizer, threshold=0, a=1e-3):
    sparse_mtrx = vectorizer.fit_transform(sentences)
    weights = sparse_mtrx.sum(axis=0)
    total   = sparse_mtrx.sum()
    weights = a / (a + weights/total)
    return weights, vectorizer

def s3e(sentence, model, tokenizer, weights, vectorizer):
    input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    tokens = tokenizer.tokenize(sentence)
    token2id = vectorizer.vocabulary_
    sample_weights = np.array([weights[0, token2id[token]] for token in tokens])
    with torch.no_grad():
        scores, hiddens = self.model(input['input_ids'].to(device), output_hidden_states=True)
        embedding = hiddens[0].squeeze().cpu().numpy()
    
    clusterer = Kmeans().fit(embeddings, sample_weights=sample_weights)
    centers, labels = clusterer.cluster_centers_, clusterer.lables_
    semantic_matrix = []
    for i in range(centers.shape[0]):
        points = embeddings[[labels == i], :]
        nth_token = np.nonzero(labels==i)
        weight_vector = sample_weights[nth_token]
        v_i = np.matmul(weight_vector, (points - centers[i:i+1, :]))
        semantic_matrix.append(v_i)
    semantic_matrix = np.array(semantic_matrix)
    
    