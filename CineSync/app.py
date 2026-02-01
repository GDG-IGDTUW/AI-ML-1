import streamlit as st
import pandas as pd
import utils
import uuid

# Page configuration
st.set_page_config(
    page_title="CineSync",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS for a better look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .recommendation-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .movie-title {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.title("🎬 CineSync")
    st.subheader("Discover your next favorite movie")
    st.markdown("Type in a movie you love, and we'll recommend others with similar plots and genres.")

    # Load Data
    data_path = 'data/movies_sample.csv'
    df = utils.load_data(data_path)
    ratings_path = 'data/ratings_sample.csv'
    ratings_df = pd.read_csv(ratings_path)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        
    if df is not None:
        # Preprocess and Compute Similarity Matrix
        # (In a real production app, you might cache this step or load a pre-computed model)
        df_processed = utils.preprocess_data(df)
        cosine_sim = utils.calculate_similarity(df_processed)

        # User Input
        # Create a list of titles for the selectbox/autocomplete experience
        movie_titles = df_processed['title'].tolist()
        
        # We use a selectbox here for better UX (prevents typos), 
        # but the requirements asked for "Users type in a movie". 
        # A selectbox with search capabilities is the best of both worlds.
        selected_movie = st.selectbox(
            "Select or type a movie name:",
            options=[""] + movie_titles, # Add empty option at start
            help="Start typing to search for a movie in our database."
        )

        uplifting_only = st.checkbox("Show only uplifting movies 😊")

        is_host = st.checkbox(
            "I am the host (controls play/pause)",
            key="group_watch_host_checkbox"
        )

        room_id = st.text_input(
            "Enter Group Watch Room ID",
            help="Share this Room ID with friends to watch together"
        )


        # --- Collaborative Filtering Section ---
        st.subheader("Personalized Recommendations (Collaborative Filtering)")

        user_id = st.selectbox(
            "Select User ID",
            ratings_df['userId'].unique(),
            help="Choose a user to get personalized recommendations"
        )

        if st.button("Find Similar Movies"):
            if room_id:
                utils.emit_sync_event(
                room_id=room_id,
                user_id=st.session_state.session_id,
                action="play",
                timestamp=0
                )
                
            if selected_movie:
                recommendations = utils.get_recommendations(selected_movie, df_processed, cosine_sim)
                
                if recommendations:
                    st.success(f"Movies similar to **{selected_movie}**:")
                    
                    # Display recommendations in a nice format
                    for movie in recommendations:
                        # Find the row for this movie to get genre/plot if we wanted to display it
                        # For now, just the title as per MVP
                        row = df[df['title'] == movie].iloc[0]
                        genres = row['genres']
                        plot = row['plot']

                        if uplifting_only and not utils.is_uplifting(plot):
                            continue
                        
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="movie-title">{movie}</div>
                            <div style="color: #666; font-size: 0.9em; margin-top: 5px;">🎭 {genres}</div>
                            <div style="margin-top: 10px; font-style: italic;">"{plot}"</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found. Try another movie!")
            else:
                st.error("Please select a movie first.")

            # --- Collaborative Filtering Button ---
            recs = None
            if st.button("Get Personalized Recommendations"):
                recs = utils.collaborative_recommendations(
                ratings_df,
                user_id=user_id,
                top_n=5
            )

            if recs:
                st.success("Recommended for you:")
                for movie in recs:
                    st.write("🎬", movie)
             
        # Show dataset stats (optional, good for transparency)
        with st.expander("See Dataset used"):
            st.dataframe(df.style.highlight_max(axis=0))

    else:
        st.error("Could not load the dataset. Please ensure 'data/movies_sample.csv' exists.")

if __name__ == '__main__':
    main()
