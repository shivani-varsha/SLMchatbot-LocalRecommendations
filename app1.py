from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from rapidfuzz import fuzz, process
import spacy
import re
from datetime import datetime
import os
import numpy as np

app = Flask(__name__)

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load the dataset
df = pd.read_csv('The_final_data.csv')

# Ensure datetime conversion for the entire DataFrame at the start
df['from_DepartureTime'] = pd.to_datetime(df['from_DepartureTime'], format='%H:%M', errors='coerce')

def get_similar_stations(df, station_name, threshold):
    stations = df['from_StationName'].unique().tolist() + df['to_StationName'].unique().tolist()
    matched_stations = set()
    results = process.extract(station_name, stations, scorer=fuzz.partial_ratio, score_cutoff=threshold)
    for result in results:
        matched_stations.add(result[0].strip())
    return list(matched_stations)

def filter_trains(df, start_station, end_station, departure_time=None, day_of_week=None, min_day_count=None, threshold=40):
    similar_start_stations = get_similar_stations(df, start_station, threshold)
    similar_end_stations = get_similar_stations(df, end_station, threshold)
    filtered_df = df[(df['from_StationName'].isin(similar_start_stations)) & (df['to_StationName'].isin(similar_end_stations))]

    if day_of_week:
        day_column = f'trainRunsOn{day_of_week}'
        filtered_df = filtered_df[filtered_df[day_column] == 'Y']
    
    if departure_time:
        departure_time = datetime.strptime(departure_time, '%H:%M').time()
        filtered_df = filtered_df.dropna(subset=['from_DepartureTime'])
        filtered_df['time_diff'] = filtered_df['from_DepartureTime'].apply(lambda x: abs((datetime.combine(datetime.today(), x.time()) - datetime.combine(datetime.today(), departure_time)).total_seconds()) if pd.notnull(x) else float('inf'))
        filtered_df = filtered_df.sort_values('time_diff')
    
    if min_day_count is not None:
        filtered_df = filtered_df[filtered_df['dayCount'] >= min_day_count]
    
    return filtered_df.reset_index(drop=True)

def extract_station_names(question):
    doc = nlp(question)
    proper_nouns = [token.text.upper() for token in doc if token.pos_ == "PROPN"]
    if len(proper_nouns) < 2:
        return None, None
    return proper_nouns[0], proper_nouns[1]

def parse_question(question):
    question = question.lower()
    start_station = None
    end_station = None
    departure_time = None
    min_day_count = None
    day_of_week = None

    day_map = {
        "sunday": "Sun",
        "monday": "Mon",
        "tuesday": "Tue",
        "wednesday": "Wed",
        "thursday": "Thu",
        "friday": "Fri",
        "saturday": "Sat"
    }

    for day in day_map.keys():
        if day in question:
            day_of_week = day_map[day]
            break

    time_match = re.search(r'\b\d{1,2}[:.]\d{2}\b', question)
    if time_match:
        departure_time = time_match.group().replace('.', ':')

    day_count_match = re.search(r'\bminimum\s+(\d+)\s+days\b', question)
    if day_count_match:
        min_day_count = int(day_count_match.group(1))

    try:
        start_station = question.split("from")[1].split("to")[0].strip().upper()
        end_station = question.split("to")[1].split("by")[0].strip().upper()
    except IndexError:
        pass

    if not start_station or not end_station:
        words = question.split()
        if "from" in words and "to" in words:
            from_index = words.index("from") + 1
            to_index = words.index("to") + 1
            if from_index < len(words) and to_index < len(words):
                start_station = words[from_index].strip().upper()
                end_station = words[to_index].strip().upper()

    if not start_station or not end_station:
        start_station, end_station = extract_station_names(question)

    return start_station, end_station, departure_time, day_of_week, min_day_count

def get_live_train_status_link(train_number):
    today_date = datetime.today().strftime('%Y%m%d')
    base_url = "https://runningstatus.in/status/"
    return f"{base_url}{train_number}-on-{today_date}"

def get_map_train_status_link(train_number):
    today_date = datetime.today().strftime('%Y%m%d')
    base_url = "https://runningstatus.in/status/"
    return f"{base_url}{train_number}-on-{today_date}/map"

def get_most_similar_station(model, station_embeddings, station_name, top_k=1):
    query_embedding = model.encode(station_name)
    distances = np.dot(list(station_embeddings.values()), query_embedding)
    closest_indices = np.argsort(distances)[-top_k:]
    closest_stations = [list(station_embeddings.keys())[i] for i in closest_indices]
    return closest_stations

def get_trains_for_question(model, station_embeddings, question):
    start_station, end_station, departure_time, day_of_week, min_day_count = parse_question(question)
    
    if not start_station or not end_station:
        word="error--stations"
        return(word)
    # Identify the most similar stations using the model
    start_station = get_most_similar_station(model, station_embeddings, start_station)[0]
    end_station = get_most_similar_station(model, station_embeddings, end_station)[0]

    # Filter the dataset to find relevant trains
    filtered_df = filter_trains(df, start_station, end_station, departure_time, day_of_week, min_day_count)

    if filtered_df.empty:
        return {"status": "error", "message": "No trains found for the specified route."}

    filtered_df['from_DepartureTime'] = pd.to_datetime(filtered_df['from_DepartureTime'], format='%H:%M', errors='coerce')

    recommendations = []
    for _, row in filtered_df.iterrows():
        train_info = {
            "train_name": row['trainName'],
            "train_number": row['trainNumber'],
            "departure_time": row['from_DepartureTime'].strftime('%H:%M') if pd.notnull(row['from_DepartureTime']) else "N/A",
            "arrival_time": row['to_ArrivalTime'],
            "distance": int(row['totalDistance']),  # Ensure totalDistance is cast to int
            "day_count": int(row['dayCount']),      # Ensure dayCount is cast to int
            "live_status_link": get_live_train_status_link(row['trainNumber']),
            "map_status_link": get_map_train_status_link(row['trainNumber'])
        }
        recommendations.append(train_info)

    return {"status": "success", "recommendations": recommendations}

def create_input_examples(df):
    input_examples = []
    for _, row in df.iterrows():
        input_text = f"from {row['from_StationName']} to {row['to_StationName']}"
        label = float(1)
        input_examples.append(InputExample(texts=[input_text, input_text], label=label))
    return input_examples

def fine_tune_model(train_examples):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_examples, name='sts-dev')

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=1,
              evaluation_steps=100,
              warmup_steps=100,
              optimizer_params={'lr': 0.02},
              output_path='fine_tuned_model')

    return model

def get_or_train_model():
    model_path = 'fine_tuned_model'
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
        print("Loaded fine-tuned model from disk.")
    else:
        train_examples = create_input_examples(df)
        model = fine_tune_model(train_examples)
        print("Trained and saved fine-tuned model.")
    return model

# Load or train the model and compute embeddings for station names
model = get_or_train_model()
station_embeddings = {station: model.encode(station) for station in df['from_StationName'].unique().tolist() + df['to_StationName'].unique().tolist()}

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    result = get_trains_for_question(model, station_embeddings, question)
    # Convert result to JSON-compatible format
    if 'recommendations' in result:
        for recommendation in result['recommendations']:
            for key, value in recommendation.items():
                if isinstance(value, pd.Timestamp):
                    recommendation[key] = value.strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
