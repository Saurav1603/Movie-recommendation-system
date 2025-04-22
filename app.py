import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity # Keep this import
import difflib
from flask import Flask, render_template, request
import gc # Import garbage collector interface

# --- Data Loading and Preprocessing (Do this ONCE when the app starts) ---
try:
    # Load the dataset
    df = pd.read_csv('movies.csv')

    # Check if required columns exist
    if not {'title', 'genres'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'title' and 'genres' columns.")

    # Select the 'genres' feature and fill missing values
    feature_column = df['genres'].fillna('')

    # Create TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Fit and transform the 'genres' feature
    # This feature_vectors matrix (N_movies x N_features) is usually much smaller
    # than the N_movies x N_movies similarity matrix and should fit in memory.
    feature_vectors = tfidf.fit_transform(feature_column)
    print(f"Shape of TF-IDF feature vectors: {feature_vectors.shape}") # Check shape

    # **REMOVED:** Do NOT calculate the full similarity matrix here

    # Get the list of all movie titles from the 'title' column
    all_movies_title_list = df['title'].tolist()

    DATA_LOADED_SUCCESSFULLY = True
    print("Movie data loaded and TF-IDF vectors created successfully.")
    print(f"Number of movies loaded: {len(all_movies_title_list)}")

except FileNotFoundError:
    DATA_LOADED_SUCCESSFULLY = False
    print("Error: 'movies.csv' not found. Please ensure the file is in the correct directory.")
except MemoryError as me:
     DATA_LOADED_SUCCESSFULLY = False
     print(f"MemoryError during initial data loading/TF-IDF creation: {me}")
     print("Consider using a machine with more RAM or reducing the dataset size.")
except Exception as e:
    DATA_LOADED_SUCCESSFULLY = False
    print(f"Error loading or processing data during startup: {e}")


# --- Flask Application ---
app = Flask(__name__)

# --- Recommendation Function ---
def get_recommendations(favourite_movie_name, num_recommendations=10):
    """Finds movies similar to the input movie based on genres, with improved matching."""

    if not DATA_LOADED_SUCCESSFULLY:
        return None, "Error: Could not load movie data during startup. Check server logs.", []

    # --- Improved Matching Logic ---
    close_match = None
    input_lower = favourite_movie_name.lower().strip() # Normalize user input

    # 1. Check for exact case-insensitive match first
    # Create a temporary lowercase list for efficient checking (or iterate)
    titles_lower = [title.lower() for title in all_movies_title_list]
    try:
        # Find the index in the lowercase list
        match_index_lower = titles_lower.index(input_lower)
        # Get the original title using the found index
        close_match = all_movies_title_list[match_index_lower]
        print(f"Found exact case-insensitive match: {close_match}")
    except ValueError:
        # No exact match found, proceed to difflib
        pass

    # 2. If no exact match, use difflib
    if not close_match:
        print(f"No exact match for '{favourite_movie_name}', using difflib...")
        # Use the original input with difflib, as it handles case somewhat internally,
        # but compare against the original list to retrieve the correct title case.
        # Lower the cutoff slightly to be more lenient. Adjust n=3 to get a few options.
        movie_recommendation_matches = difflib.get_close_matches(
            favourite_movie_name, # Use original input for difflib's comparison
            all_movies_title_list,
            n=1, # Get the single best match
            cutoff=0.5 # Lowered threshold (default is 0.6, experiment with this value)
        )

        if not movie_recommendation_matches:
            # Still no match found even with lower cutoff
            return None, f"Sorry, couldn't find a close match for '{favourite_movie_name}'. Please try another movie title.", []

        close_match = movie_recommendation_matches[0]
        print(f"Found close match via difflib: {close_match}")

    # --- Proceed with recommendation using the found close_match ---

    # Find the index of the closest match movie in the original DataFrame
    try:
        # Use the 'title' column and get the actual DataFrame index
        index_of_close_match_movie = df[df.title == close_match].index[0]
    except IndexError:
         # This could happen if a title exists in all_movies_title_list but somehow
         # isn't found in the DataFrame lookup (e.g., data cleaning issues).
         return close_match, f"Error: Could not find index for the matched movie '{close_match}'. Data inconsistency?", []

    # **MODIFIED: Calculate similarity ON DEMAND**
    try:
        # Get the TF-IDF vector for the input movie using its DataFrame index
        input_movie_vector = feature_vectors[index_of_close_match_movie]

        # Calculate cosine similarity between the input movie's vector and ALL movie vectors
        # Result shape: (1, N_movies). We take the first (and only) row [0].
        recommendation_scores_vector = cosine_similarity(input_movie_vector, feature_vectors)[0]

        # Create a list of (index, score) tuples
        recommendation_scores = list(enumerate(recommendation_scores_vector))

        # Sort movies based on similarity score (descending)
        sorted_similar_movies = sorted(recommendation_scores, key=lambda x: x[1], reverse=True)

        # Clean up large intermediate objects if needed
        del recommendation_scores_vector
        gc.collect()

    except MemoryError:
        print(f"MemoryError occurred while calculating similarity for '{close_match}'.")
        return close_match, "Error: Ran out of memory while calculating recommendations. The server might be overloaded.", []
    except Exception as e:
         print(f"Error during similarity calculation or sorting for '{close_match}': {e}")
         return close_match, "Error: An unexpected issue occurred while finding recommendations.", []


    # Get the top N recommended movie titles (excluding the input movie itself)
    recommended_movies = []
    i = 1
    for movie_index, score in sorted_similar_movies:
        # Skip the input movie itself (it will always have the highest score of 1.0)
        if movie_index == index_of_close_match_movie:
            continue

        # Use .iloc for efficient index-based lookup in the DataFrame
        try:
            title_from_index = df.iloc[movie_index]['title']
            recommended_movies.append(title_from_index)
            i += 1
            # Stop once we have enough recommendations
            if i > num_recommendations:
                break
        except IndexError:
            # This shouldn't happen if indices from enumerate are correct, but safety check
            print(f"Warning: Index {movie_index} out of bounds when retrieving title.")
            continue # Skip this movie if index is bad


    if not recommended_movies:
         # This might happen if all other movies have a similarity score of 0
         return close_match, f"Found match '{close_match}', but no similar movies found based on genres.", []

    # Return the matched movie title, no error, and the list of recommendations
    return close_match, None, recommended_movies


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error_message = None
    movie_input = ""
    close_match_found = None

    # Check if data failed to load during startup
    if not DATA_LOADED_SUCCESSFULLY:
         error_message = "Application Error: Movie data could not be loaded during startup. Please check server logs."
         return render_template('index.html',
                                error_message=error_message,
                                recommendations=[],
                                movie_input="",
                                close_match=None)

    # Handle POST request (user submitted the form)
    if request.method == 'POST':
        movie_input = request.form.get('movie_name', '').strip()
        if movie_input:
            # Call the recommendation function
            close_match_found, error_message, recommendations = get_recommendations(movie_input)
        else:
            # Handle empty input
            error_message = "Please enter a movie name."

    # Pass data to the HTML template (for both GET and POST requests)
    return render_template('index.html',
                           recommendations=recommendations,
                           error_message=error_message,
                           movie_input=movie_input,       # Send back the user's input
                           close_match=close_match_found) # Send back the title that was actually matched

# --- Run the App ---
if __name__ == '__main__':
    # Set debug=False for production environment
    # Use host='0.0.0.0' to make it accessible on your network if needed
    app.run(debug=True) # Keep debug=True for development testing