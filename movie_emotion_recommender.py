# ---------------------------------------------
# movie_emotion_recommender.py
# ---------------------------------------------

import pandas as pd
import ast
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings('ignore')

def main():
    # ------------------------------
    # 1️⃣ Charger le dataset TMDB
    # ------------------------------
    url = "https://raw.githubusercontent.com/vamshi121/TMDB-5000-Movie-Dataset/master/tmdb_5000_movies.csv"
    df = pd.read_csv(url)
    print(f"Dataset chargé ! Dimensions : {df.shape}")

    # ------------------------------
    # 2️⃣ Parser la colonne 'genres'
    # ------------------------------
    def extract_genres(genres_str):
        try:
            genres = ast.literal_eval(genres_str)
            return [g["name"] for g in genres]
        except:
            return []

    df["genres_list"] = df["genres"].apply(extract_genres)

    # ------------------------------
    # 3️⃣ Mapping : Genre → Emotions
    # ------------------------------
    genre_to_emotions = {
        "Action": ["excitation", "colère"],
        "Adventure": ["excitation", "joie"],
        "Animation": ["joie", "amusement"],
        "Comedy": ["joie", "amusement"],
        "Crime": ["tension", "stress"],
        "Documentary": ["curiosité", "réflexion"],
        "Drama": ["tristesse", "réflexion"],
        "Family": ["joie", "sérénité"],
        "Fantasy": ["émerveillement", "joie"],
        "History": ["réflexion", "nostalgie"],
        "Horror": ["peur", "tension"],
        "Music": ["joie", "enthousiasme"],
        "Mystery": ["curiosité", "tension"],
        "Romance": ["amour", "nostalgie"],
        "Science Fiction": ["émerveillement", "excitation"],
        "TV Movie": ["divertissement", "joie"],
        "Thriller": ["tension", "stress"],
        "War": ["colère", "tristesse"],
        "Western": ["aventure", "réflexion"]
    }

    def emotions_from_genres(genres):
        emos = []
        for g in genres:
            emos += genre_to_emotions.get(g, [])
        return list(set(emos))

    df["emotions_from_genres"] = df["genres_list"].apply(emotions_from_genres)

    # ------------------------------
    # 4️⃣ Chargement modèle NLP anglais
    # ------------------------------
    print("Chargement du modèle NLP anglais (distilroberta)...")
    emotion_analyzer_en = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

    # ------------------------------
    # 5️⃣ Détection émotion + traduction FR→EN
    # ------------------------------
    def get_emotion_from_text(text):
        if pd.isna(text) or text.strip() == "":
            return []

        try:
            # Détecter langue
            lang = detect(text)
            if lang == 'fr':
                text = GoogleTranslator(source='fr', target='en').translate(text)

            # Analyse émotion anglaise
            result = emotion_analyzer_en(text)[0]
            # Trier par score décroissant et prendre top 2 émotions
            result = sorted(result, key=lambda x: x["score"], reverse=True)
            return [r["label"] for r in result[:2]]
        except:
            return []

    print("Analyse des overviews...")
    df["emotion_overview"] = df["overview"].apply(get_emotion_from_text)
    print("Analyse des taglines...")
    df["emotion_tagline"] = df["tagline"].apply(get_emotion_from_text)

    # ------------------------------
    # 6️⃣ Fusionner toutes les émotions pour créer 'emotions_associees'
    # ------------------------------
    def combine_emotions(row):
        emos = set(row["emotions_from_genres"])
        emos.update(row["emotion_overview"])
        emos.update(row["emotion_tagline"])
        return list(emos)

    df["emotions_associees"] = df.apply(combine_emotions, axis=1)

    # ------------------------------
    # 7️⃣ Sauvegarder dataset enrichi
    # ------------------------------
    df.to_csv("tmdb_movies_emotions.csv", index=False)
    print("Dataset enrichi sauvegardé sous 'tmdb_movies_emotions.csv'")

    # ------------------------------
    # 8️⃣ Affichage rapide
    # ------------------------------
    cols_to_show = ["title", "genres_list", "emotions_associees"]
    print(df[cols_to_show].head(20))


if __name__ == "__main__":
    main()
