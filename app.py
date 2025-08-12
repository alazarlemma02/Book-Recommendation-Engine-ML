import streamlit as st
import pandas as pd
import pickle
import os
import random
from datetime import datetime

# ==== Load model and data ====
artifact_files = {
    "model": "artifacts/model.pkl",
    "books_pivot": "artifacts/books_pivot.pkl",
    "book_names": "artifacts/book_name.pkl",
    "final_rating": "artifacts/final_rating.pkl"
}

for name, path in artifact_files.items():
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}. Please ensure all model/data files are present.")
        st.stop()

with open(artifact_files["model"], 'rb') as f:
    model = pickle.load(f)
with open(artifact_files["books_pivot"], 'rb') as f:
    books_pivot = pickle.load(f)
with open(artifact_files["book_names"], 'rb') as f:
    book_names = pickle.load(f)
with open(artifact_files["final_rating"], 'rb') as f:
    final_rating = pickle.load(f)

# ==== Setup interaction logging ====
LOG_FILE = 'users/user_logs.csv'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["user_id", "timestamp", "action", "book_title"]).to_csv(LOG_FILE, index=False)

def log_interaction(user_id, action, book_title):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[user_id, timestamp, action, book_title]],
                         columns=["user_id", "timestamp", "action", "book_title"])
    entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

# ==== Collaborative Filtering ====
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_book_recommendations(title, n_recommendations=9):
    if title not in books_pivot.index:
        return []

    # Get the selected book info
    book_info = final_rating[final_rating['title'] == title].drop_duplicates('title').iloc[0]
    main_book = {
        "title": title,
        "author": book_info["author"],
        "image_url": book_info["image_url"]
    }

    # Get recommended similar books
    book_index = books_pivot.index.get_loc(title)
    distances, indices = model.kneighbors(
        books_pivot.iloc[book_index, :].values.reshape(1, -1),
        n_neighbors=n_recommendations + 1
    )

    recommended_books = []
    for i in range(1, len(distances.flatten())):
        book = books_pivot.index[indices.flatten()[i]]
        book_info = final_rating[final_rating['title'] == book].drop_duplicates('title').iloc[0]
        recommended_books.append({
            "title": book,
            "author": book_info["author"],
            "image_url": book_info["image_url"]
        })

    return [main_book] + recommended_books

# ==== Personalized Recommendations ====
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_user_history_recommendations(user_id, n_recommendations=15):
    if not os.path.exists(LOG_FILE):
        return []

    logs = pd.read_csv(LOG_FILE)
    user_logs = logs[logs["user_id"] == user_id]

    liked_books = user_logs[user_logs["action"] == "liked"]["book_title"].unique().tolist()
    searched_books = user_logs[user_logs["action"] == "searched"]["book_title"].unique().tolist()

    user_books = liked_books + searched_books
    user_books = list(dict.fromkeys(user_books))  # Remove duplicates while preserving order
    random.shuffle(user_books)

    seen = set()
    all_recommendations = []

    for title in user_books:
        if title in books_pivot.index:
            for rec in get_book_recommendations(title, n_recommendations):
                if rec["title"] not in seen and rec["title"] not in user_books:
                    all_recommendations.append(rec)
                    seen.add(rec["title"])
                if len(all_recommendations) >= n_recommendations:
                    return all_recommendations

    return all_recommendations

# ==== Simple Login ====
st.session_state.setdefault("authenticated", False)

if not st.session_state["authenticated"]:
    st.title("🔐 Login to Book Recommender")
    username = st.text_input("Enter your username:")
    if username:
        st.session_state["authenticated"] = True
        st.session_state["user_id"] = username
        st.rerun()
else:
    user_id = st.session_state["user_id"]
    st.sidebar.success(f"👤 Logged in as: {user_id}")
    st.title("Smart Book Recommender")

    # === Search area on top ===
    st.subheader("Search for a book you like:")
    search_input = st.text_input("Start typing a book title:")

    if search_input:
        filtered_books = [book for book in book_names if search_input.lower() in book.lower()]
        if filtered_books:
            selected_book = st.selectbox("Select a book", filtered_books)
            log_interaction(user_id, "searched", selected_book)

            st.subheader(f"Recommendations based on: {selected_book}")
            recommendations = get_book_recommendations(selected_book)

            if recommendations:
                cols = st.columns(3)
                for idx, book in enumerate(recommendations):
                    with cols[idx % 3]:
                        st.image(book["image_url"], width=120)
                        st.markdown(f"**{book['title']}**")
                        st.markdown(f"*{book['author']}*")
                        if st.button("👍 Like", key=f"like_{book['title']}_search"):
                            log_interaction(user_id, "liked", book["title"])
            else:
                st.warning("No recommendations found for this book.")
        else:
            st.warning("No matching books found.")

    # === Personalized recommendations ===
    st.subheader("Recommended for You")
    recs = get_user_history_recommendations(user_id)

    if recs:
        cols = st.columns(3)
        for idx, book in enumerate(recs):
            with cols[idx % 3]:
                st.image(book["image_url"], width=120)
                st.markdown(f"**{book['title']}**")
                st.markdown(f"*{book['author']}*")
                if st.button("👍 Like", key=f"like_{book['title']}_rec"):
                    log_interaction(user_id, "liked", book["title"])
    else:
        st.info("Start searching to receive personalized recommendations!")
