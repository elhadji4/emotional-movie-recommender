# ---------------------------------------------
# emotion_chatbot.py
# ---------------------------------------------

import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
from transformers import pipeline
import langdetect
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1️⃣ Charger le dataset enrichi
# ------------------------------
df_movies = pd.read_csv("tmdb_movies_emotions.csv")

# Créer un dictionnaire Emotion -> Films
emotion_to_movies = {}
for _, row in df_movies.iterrows():
    for emo in row['emotions_associees'].strip("[]").replace("'", "").split(", "):
        if emo not in emotion_to_movies:
            emotion_to_movies[emo] = []
        emotion_to_movies[emo].append(row['title'])

# ------------------------------
# 2️⃣ Charger les modèles NLP
# ------------------------------
print("Chargement du modèle NLP anglais (distilroberta)...")
model_en = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Si tu veux activer le français, vérifie le modèle dispo sur HuggingFace
# Pour l'instant on garde juste l'anglais pour éviter les erreurs
model_fr = None

# ------------------------------
# 3️⃣ Emojis pour chaque émotion
# ------------------------------
emotion_emojis = {
    "joie": "😄",
    "tristesse": "😢",
    "colère": "😡",
    "peur": "😱",
    "réflexion": "🤔",
    "amusement": "😂",
    "excitation": "🤩",
    "tension": "😬",
    "stress": "😣",
    "curiosité": "🧐",
    "sérénité": "😌",
    "amour": "😍",
    "nostalgie": "😔",
    "divertissement": "🎉",
    "émerveillement": "✨",
    "enthousiasme": "😁",
    "aventure": "🏕️",
    "neutre": "😐",
    "dégoût": "🤢"
}

# ------------------------------
# 4️⃣ Détecter l’émotion
# ------------------------------
def detect_emotion(text):
    if not text.strip():
        return "neutre", 0.0
    try:
        lang = langdetect.detect(text)
    except:
        lang = "en"
    try:
        if lang == "fr" and model_fr:
            result = model_fr(text)[0]
        else:
            result = model_en(text)[0]
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        return result[0]["label"], result[0]["score"]
    except:
        return "neutre", 0.0

# ------------------------------
# 5️⃣ Recommandation
# ------------------------------
def recommend_movies(emotion):
    return emotion_to_movies.get(emotion.lower(), ["Aucun film trouvé"])

# ------------------------------
# 6️⃣ Interface Tkinter
# ------------------------------
def send_message():
    user_text = entry.get()
    if not user_text.strip():
        return
    chat_window.config(state='normal')
    chat_window.insert(tk.END, f"🧑 Toi : {user_text}\n")
    
    # Détecter l'émotion
    emotion, confidence = detect_emotion(user_text)
    emoji = emotion_emojis.get(emotion, "😐")
    movies = recommend_movies(emotion)
    
    # Afficher la réponse du bot
    response = f"🤖 Chatbot : Je détecte que tu te sens **{emotion}** {emoji} (confiance: {confidence:.2f})\n\n"
    response += "🎬 Films recommandés :\n - " + "\n - ".join(movies[:10]) + "\n\n"
    chat_window.insert(tk.END, response)
    chat_window.yview(tk.END)
    chat_window.config(state='disabled')
    entry.delete(0, tk.END)

# ------------------------------
# Fenêtre principale
# ------------------------------
root = tk.Tk()
root.title("🎭 Emotional Movie Recommender")
root.geometry("650x650")
root.resizable(False, False)

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Helvetica", 12))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# ✅ Message de bienvenue automatique
chat_window.config(state='normal')
chat_window.insert(tk.END, "🤖 Chatbot : Salut ! Comment s'est passée ta journée ? / Hey, how was your day?\n\n")
chat_window.config(state='disabled')

entry_frame = tk.Frame(root)
entry_frame.pack(fill=tk.X, padx=10, pady=5)

entry = tk.Entry(entry_frame, font=("Helvetica", 14))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
entry.bind("<Return>", lambda event: send_message())

send_button = tk.Button(entry_frame, text="Envoyer", font=("Helvetica", 12), command=send_message)
send_button.pack(side=tk.RIGHT)

root.mainloop()
