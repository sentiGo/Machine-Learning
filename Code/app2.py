#!/usr/bin/env python
# coding: utf-8

# In[11]:


from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np

app = Flask(__name__)

#load the model
model = tf.keras.models.load_model('./model_recommendation.h5')

#load the dataset
data = pd.read_csv('./data/Datasets_filtering_2.csv')
text_data = data['description'].tolist()

#inisialize tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data['description'].tolist())
max_len = model.input_shape[1]

#tokenize and pad all sequences
sequences = tokenizer.texts_to_sequences(text_data)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

#cosine_similarity function
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

@app.route('/predict', methods=['POST'])
def predict():
    #get user input from request body
    user_input = request.args.get('text')
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(user_sequence, maxlen=max_len)
    
    #get the user input embedding
    user_embedding = model.predict(user_padded_sequence)[0]
    
    #Calculate similarity
    embeddings = model.predict(padded_sequences)
    similarity_scores = []
    for embedding in embeddings:
        similarity_scores.append(cosine_similarity(user_embedding, embedding))
    
    #get the most similarity from data records
    top_records_indices = np.argsort(similarity_scores)[-5:]
    top_records = data.loc[top_records_indices]
    top_scores = [similarity_scores[i] for i in top_records_indices]
    
    recommendations = []
    for i in range(len(top_records)):
        recommendation = {
            'Record': int(top_records_indices[i])+1,
            'Similarity Score': float(top_scores[i]),
            'Data': top_records.iloc[i].to_dict()
        }
        recommendations.append(recommendation)
    return jsonify(recommendations)

if __name__== '__main__':
    app.run(port='8080')

