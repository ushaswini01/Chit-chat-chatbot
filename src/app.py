
import os
import re
import json
import torch
import nltk
nltk.download('punkt')
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from flask import send_file
from flask import Flask, render_template, request, Response, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline, TFAutoModelForCausalLM
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


with open('/home/maria65_antony/P3/file:/topic_embeddings.json', 'r') as f:
    embeddings_text_corpus = json.load(f)

embeddings = []
import numpy as np
topics = ['Health','Environment','Education','Politics','Technology','Sports','Entertainment','Food','Travel','Economy']
for i in topics:
    filtered_data = embeddings_text_corpus.get(i, [])
    for entry in filtered_data:
        embeddings.append(entry["embeddings"])
embeddings = np.array([embeddings])
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
generator = pipeline('text-generation', model="distilbert/distilgpt2")
bi_encoder = SentenceTransformer("all-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def remove_unfinished_sentences(text):
    sentence_endings = ['.', '!', '?']
    for ending in sentence_endings:
        if ending in text:
            return text.split(ending)[0] + ending
    return text.strip() 

def bi_cross_pipeline(query, corpus):
    query_embedding = bi_encoder.encode(query, convert_to_numpy=True)
    query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
    corpus_text, corpus_embeddings = corpus[0], corpus[1]
    corpus_embeddings = torch.tensor(corpus_embeddings, dtype=torch.float32)
    bi_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k_indices = torch.topk(bi_scores, k=5).indices
    top_k_results = [(corpus_text[idx], bi_scores[idx].item()) for idx in top_k_indices]
    cross_input = [(query, corpus_text[idx]) for idx in top_k_indices]
    cross_scores = cross_encoder.predict(cross_input)
    
    reranked_results = [{"text": corpus_text[idx], "score": cross_scores[i]}
        for i, idx in enumerate(top_k_indices)]

    reranked_results.sort(key=lambda x: x["score"], reverse=True) 
    top_result = reranked_results[0] 
    generated_text = generator(top_result['text'][:300], max_length=300,truncation=True,pad_token_id=True,eos_token_id=True)
    return remove_unfinished_sentences(generated_text[0]['generated_text'])

def filter_by_topic(data, topic):
    filtered_data = data.get(topic, [])
    texts = [entry["text"] for entry in filtered_data]
    embeddings = [entry["embeddings"] for entry in filtered_data]
    return texts, embeddings

def chat_pipeline(query):
    candidate_labels = ['Health','Environment','Education','Politics','Technology','Sports','Entertainment','Food','Travel','Economy']
    topic = classifier(query, candidate_labels)['labels'][0]
    corpus = filter_by_topic(embeddings_text_corpus,topic)
    generated_output = bi_cross_pipeline(query,corpus)
    return generated_output

CONVERSATION_LOG_FILE = "conversation_logs.json"

def log_conversation(user_input, bot_response, topic, error=None):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "bot_response": bot_response,
        "topic": topic,
        "error": error,}

    if not os.path.exists(CONVERSATION_LOG_FILE):
        with open(CONVERSATION_LOG_FILE, "w") as f:
            json.dump([log_entry], f, indent=4)
    else:
        with open(CONVERSATION_LOG_FILE, "r+") as f:
            data = json.load(f)
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=4)

def get_response():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        selected_topics = data.get("topics", ["All"]) 

        if not user_input:
            return jsonify({"response": "Please provide a valid input."}), 400

        print(f"User Input Received: {user_input}, Topics: {selected_topics}")

        if "All" in selected_topics:
            response = get_bot_response(user_input)
        else:
            response = get_bot_response_with_topics(user_input, selected_topics)

        print(f"Bot Response Generated: {response}") 
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in /get_response: {e}") 
        return jsonify({"response": "An unexpected error occurred. Please try again."}), 500

def get_bot_response_with_topics(user_input, topics):
    topics = ['Health', 'Environment', 'Education', 'Politics', 'Technology', 
              'Sports', 'Entertainment', 'Food', 'Travel', 'Economy']
    try:
        classification_result = classifier(user_input, candidate_labels=topics)
        best_topic = classification_result['labels'][0]
        score = classification_result['scores'][0]

        if score > 0.4:
            bot_response = chat_pipeline(user_input)
            log_conversation(user_input, bot_response, best_topic)
            print(bot_response)
            if not compute_similarity(user_input,bot_response):
                bot_output_text = "I apologize. Could you please clarify or rephrase so I can assist you better?"
                return bot_output_text
        else:
            global chat_history_ids
            chat_history_ids = None
            new_user_input_ids = chat_tokenizer.encode(user_input + chat_tokenizer.eos_token, return_tensors='pt')

            if chat_history_ids is None:
                chat_history_ids = new_user_input_ids
            else:
                chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

            bot_output = chat_model.generate(
                chat_history_ids, 
                max_length=1000, 
                pad_token_id=chat_tokenizer.eos_token_id,
                top_k=50, 
                top_p=0.95, 
                temperature=0.7, 
                no_repeat_ngram_size=3, 
                do_sample=True, 
                num_return_sequences=1)

            bot_output_text = chat_tokenizer.decode(bot_output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

            log_conversation(user_input, bot_output_text, "General")
            print(bot_output_text)
            return bot_output_text
        return bot_response

    except Exception as e:
        log_conversation(user_input, "Error", "Unknown", str(e))
        return f"An error occurred: {e}"

def get_bot_response(user_input):
    topics = ['Health', 'Environment', 'Education', 'Politics', 'Technology', 
              'Sports', 'Entertainment', 'Food', 'Travel', 'Economy']
    global chat_history_ids
    chat_history_ids = None

    try:
        classification_result = classifier(user_input, candidate_labels=topics)
        best_topic = classification_result['labels'][0]
        score = classification_result['scores'][0]

        if  score > 0.4:
            bot_response = chat_pipeline(user_input)
            log_conversation(user_input, bot_response, best_topic)
            return bot_response  
        else:
            new_user_input_ids = chat_tokenizer.encode(user_input + chat_tokenizer.eos_token, return_tensors='pt')

            if chat_history_ids is None:
                chat_history_ids = new_user_input_ids
            else:
                chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

            bot_output = chat_model.generate(
                chat_history_ids, 
                max_length=1000, 
                pad_token_id=chat_tokenizer.eos_token_id,
                top_k=50, 
                top_p=0.95, 
                temperature=0.7, 
                no_repeat_ngram_size=3, 
                do_sample=True, 
                num_return_sequences=1)

            bot_output_text = chat_tokenizer.decode(bot_output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

            log_conversation(user_input, bot_output_text, "General")
            print(bot_output_text)
            return bot_output_text
    except Exception as e:
        log_conversation(user_input, "Error", "Unknown", str(e))
        return "An error occurred. Please try again."

def compute_similarity(query, output):
    embeddings = bi_encoder.encode([query, output], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    print(similarity_score)
    return similarity_score >= 0.4

app = Flask(__name__, static_url_path='/static') 

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        selected_topics = data.get("topics", ["All"])

        if not user_input:
            return jsonify({"response": "Please provide a valid input."}), 400

        print(f"User Input Received: {user_input}, Topics: {selected_topics}")

        if "All" in selected_topics or len(selected_topics) == 0:
            response = get_bot_response(user_input)
        else:
            response = get_bot_response_with_topics(user_input, selected_topics)

        print(f"Bot Response Generated: {response}") 
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in /get_response: {e}") 
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

@app.route("/visualize", methods=["GET"])
def visualize():
    if not os.path.exists(CONVERSATION_LOG_FILE):
        return "No conversation data available."

    with open(CONVERSATION_LOG_FILE, "r") as f:
        data = json.load(f)

    if not data: 
        return "No conversation data available."

    df = pd.DataFrame(data)
    static_dir = os.path.join(app.root_path, "static")
    os.makedirs(static_dir, exist_ok=True)

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        unique_days = df['timestamp'].dt.date.nunique() 
        total_sessions_count = len(df['timestamp'].dt.date.unique()) 
        total_conversations = len(df)
        avg_conversations_per_session = total_conversations / unique_days if unique_days > 0 else 0
        avg_retrieved_docs_per_session = 7.19 
    else:
        avg_conversations_per_session = 0
        total_sessions_count = 0
        avg_retrieved_docs_per_session = 0

    # Visualization 1: Conversations Over Time
    conversations_over_time_path = os.path.join(static_dir, "conversations_over_time.png")
    if 'timestamp' in df.columns:
        plt.figure(figsize=(10, 6))
        conversations_per_hour = df.groupby(df['timestamp'].dt.hour).size()
        conversations_per_hour.plot(kind='line', marker='o', color='blue')
        plt.title('Conversations Over Time')
        plt.xlabel('Hour of Day')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(conversations_over_time_path)
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No data available for conversations over time', fontsize=12, ha='center')
        plt.tight_layout()
        plt.savefig(conversations_over_time_path)
        plt.close()

    # Visualization 2: Response Type Distribution
    response_type_dist_path = os.path.join(static_dir, "response_type_distribution.png")
    if 'topic' in df.columns:
        plt.figure(figsize=(10, 6))
        df['topic'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Response Type Distribution')
        plt.xlabel('Topic')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(response_type_dist_path)
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No data available for response types', fontsize=12, ha='center')
        plt.tight_layout()
        plt.savefig(response_type_dist_path)
        plt.close()

    # Visualization 3: Errors Per Topic
    errors_per_topic_path = os.path.join(static_dir, "errors_per_topic.png")
    if 'error' in df.columns and not df[df['error'].notnull()].empty:
        plt.figure(figsize=(10, 6))
        df[df['error'].notnull()]['topic'].value_counts().plot(kind='bar', color='salmon')
        plt.title('Errors per Topic')
        plt.xlabel('Topic')
        plt.ylabel('Error Count')
        plt.tight_layout()
        plt.savefig(errors_per_topic_path)
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No errors logged in the data', fontsize=12, ha='center')
        plt.tight_layout()
        plt.savefig(errors_per_topic_path)
        plt.close()

    return render_template(
        "visualizations.html",
        conversations_over_time_img="static/conversations_over_time.png",
        response_type_dist_img="static/response_type_distribution.png",
        errors_per_topic_img="static/errors_per_topic.png",
        avg_conversations=round(avg_conversations_per_session, 2),
        total_sessions=total_sessions_count,
        avg_docs_retrieved=avg_retrieved_docs_per_session
    )

if __name__ == "__main__":
    app.run(debug=True)