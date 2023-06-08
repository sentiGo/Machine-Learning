#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install flask')

from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np

app = Flask(__name__)

#load the model
model = tf.keras.models.load_model('./model_recommendation.h5')

#load the dataset
data = pd.read_csv('./data/Datasets_filtering_2.csv')

#inisialize tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data['description'].tolist())

#cosine_similarity function
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

@app.route('/prediction_similarity', methods=['POST'])
def prediction_similarity():
    #get user input from request body
    user_input = request.json['input']
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(user_sequence, maxlen=model.input_shape[1])
    
    #get the user input embedding
    user_embedding = model.predict(user_padded_sequence)[0]
    
    #Calculate similarity
    embeddings = model.predict(model.padded_sequences)
    similarity_scores = []
    for embedding in embeddings:
        similarity_scores.append(cosine_similarity(user_embedding, embedding))
    
    #get the most similarity from data records
    top_records_indices = np.argsort(similarity_scores)[-5:]
    top_records = data.iloc[top_records_indices].to_dict(orient='records')
    return jsonify(top_records)

if __name__== '__main__':
    app.run()





