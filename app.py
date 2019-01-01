# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:21:38 2019

@author: WELCOME
"""

from gensim.models.doc2vec import Doc2Vec


import gensim
from gensim.test.utils import datapath
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import gensim
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))




from flask import Flask, redirect, url_for, request,jsonify
app = Flask(__name__)

model = gensim.models.Doc2Vec.load('C:\\Users\\WELCOME\\Desktop\\Traning\\d2v.model') 

def get_sentences_label():
    text = []
    texts = []
    filename ='C:\\Users\\WELCOME\Desktop\\Traning\\diabetes&spinal.txt'
    with open(filename) as f:
        text.append(f.readlines())
        
        text = text[0]

    for text in text:
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        texts.append(filtered_sentence)
        filtered_list = []
    for text in texts:
        filtered_list.append(" ".join(text))
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i).lower()]) for i, _d in enumerate(filtered_list)]
    return tagged_data
    

def normalize_similarity(similar_list_of_tuples):
    document_collection = []
    for doc_id,similarity in similar_list_of_tuples:
        document_collection.append(int(doc_id))
        return document_collection
    


def remove_special_characters(similar_documents):
    top_3_documents = []
    for documents in similar_documents:
        top_3_documents.append(' '.join(e for e in documents if e.isalnum()))
    return top_3_documents

@app.route('/similar', methods=['GET', 'POST'])
def similar_phrase(tagged_data=get_sentences_label()):
    content = request.json
    phrase = content['phrase']
    phrase = phrase.lower()
    similar_phrase = model.docvecs.most_similar(positive=[model.infer_vector(phrase)],topn=3)
    doc_ids = normalize_similarity(similar_phrase)
    list_of_documents = []
    for doc_id in doc_ids:
        list_of_documents.append(doc_id)
    sentence_list = []
    for doc in list_of_documents: 
        sentence_list.append(tagged_data[doc][0])
    return jsonify({"similar":" ".join(sentence_list[0])})

if __name__ == '__main__':
    app.run(host= '127.0.0.1',port=80,debug=True)
        
    
        
        
        
        
    
    