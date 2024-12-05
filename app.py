import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from lyricsgenius import Genius
import time
from mpl_toolkits.mplot3d import Axes3D
from openai import OpenAI
import random

# Function to run the app
def run():
    # Create empty lists to hold the artist and song names
    artist_list = []
    song_list = []
    
    # Title for the input section
    st.title("What's Your Lyrical Map?")

    st.write("""
             To ensure that this application runs properly, please ensure that your song and arist names are spelled correctly. Additionally, please pick songs that have more than 12 unique lines.
             """)

    # First, ask the user for 5 artist and song names. If needed, we could change this to something else
    for i in range(1, 6):
        artist_name = st.text_input(f"Enter name of artist {i}:", key=f"artist_{i}")
        if artist_name:
            artist_list.append(artist_name)

        # Input for song name for the given artist
        song_name = st.text_input(f"Enter song for {artist_name}:", key=f"song_{i}")
        if song_name:
            song_list.append(song_name)

    # Finally, let the user determing the number of clusters, from 2 to 5
    num_clusters = st.number_input("Enter the number of custom genres you would like to use: ", min_value=2, max_value=5, value=3, step=1)
    
    # Submit button to process the data
    if st.button("Submit"):
        if len(artist_list) == 5 and len(song_list) == 5:
            #Initial setup 
            st.write("Set up complete. Now starting the process.")
            #print("Hello!")
            #st.write("Processing lyrics and visualizing...")
            #process_and_visualize(artist_list, song_list, num_clusters)
            backend(artist_list, song_list, num_clusters)
        else:
            st.warning("Please enter both artist and song names for all 5 artists.")

#Function that runs the whole backend. I wanted to have a clear separation between the two
def backend(artist_list, song_list, num_clusters):
    # Grab the model from HuggingFace
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        "dunzhang/stella_en_400M_v5",
        trust_remote_code=True,
        device=device,
        config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
    )

    #Get the lyrics
    with st.spinner("Fetching Lyrics..."):
        all_lyrics, all_songs = extract_lyrics(artist_list, song_list)

    #Creates the embeddings
    with st.spinner("Generating embeddings..."):
        embeddings = model.encode(all_lyrics)

    #Creates the T-SNE transformation
    with st.spinner("Performing T-SNE transformation..."):
        # T-SNE Transform and Visualization
        tsne = TSNE(n_components=3, random_state=0, perplexity=5)
        embeddings_tsne = tsne.fit_transform(embeddings)

    with st.spinner("Plotting the base visualization..."):
        plot_tsne(embeddings_tsne, all_lyrics, all_songs, "T-SNE Visualization (No Clustering)")
    
    with st.spinner("Applying KMeans Clustering..."):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans_labels = kmeans.fit_predict(embeddings_tsne)

    with st.spinner("Plotting the base visualization with clustering..."):
        plot_tsne_with_kmeans(embeddings_tsne, kmeans_labels, all_lyrics, num_clusters)
        
    with st.spinner("Creating the custom genres and displaying the list..."):
        custom_clusters(num_clusters, embeddings_tsne, kmeans_labels, all_lyrics, all_songs)

#Function to extract lyrics
def extract_lyrics(artist_list, song_list):
    #Sets up the values to return
    all_lyrics = []
    all_songs = []

    #Sets up the genius api
    Access_Token = st.secrets["LYRICS_TOKEN"]
    genius = Genius(Access_Token, timeout=30)  # Increase timeout to 30 seconds

    #Goes to every artist
    for i in range(len(artist_list)):
        #Spinner to show progress
        with st.spinner(f'{i + 1} / 5'):
            #Searches for the artist and the song
            artist = genius.search_artist(artist_list[i], max_songs=3, sort="title")
            song = genius.search_song(song_list[i], artist.name)
            lyrics = song.lyrics
            #Convers the lyric into a line by line
            lyric_list = lyrics.split("\n")
            #Filters the lyrics to clean it up a bit
            filtered_lyrics = lyric_list[1:-1]
            filtered_lyrics = [
                lyric for lyric in filtered_lyrics 
                if lyric.strip() and not (lyric.startswith('[') and lyric.endswith(']')) or "You may also like" not in lyric
                ]
            unique_lyrics = list(set(filtered_lyrics))
            random_lyrics = random.sample(unique_lyrics, 10)
            for lyric in random_lyrics:
                all_lyrics.append(lyric)
                all_songs.append(song_list[i]) 
            time.sleep(2)  #Sleep time to prevent api overload
    
    return all_lyrics, all_songs

# Function to plot T-SNE without clustering and display an index for songs
def plot_tsne(embeddings_tsne, all_lyrics, all_songs, title):
    # Define a color palette with a sufficient number of colors 
    unique_songs = list(set(all_songs))  
    colors = plt.cm.get_cmap("tab10", len(unique_songs))

    # Make sure each song has a color
    song_color_mapping = {song: colors(i) for i, song in enumerate(unique_songs)}

    # Create a list of colors for the lyrics based on the song they belong to
    lyric_colors = [song_color_mapping[song] for song in all_songs]

    # Create a figure and 3D axis for the T-SNE plot
    fig = plt.figure(figsize=(36, 24))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the T-SNE results, coloring points according to their song
    scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2], c=lyric_colors, label='Lyrics')

    # Annotate each point with its corresponding text (lyrics)
    for i, text in enumerate(all_lyrics):
        ax.text(embeddings_tsne[i, 0], embeddings_tsne[i, 1], embeddings_tsne[i, 2], text, fontsize=10)

    # Create a custom legend with song colors
    handles = []
    for song, color in song_color_mapping.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=song))

    # Display the plot
    ax.set_title(title)
    ax.legend(handles=handles, title="Songs", loc="upper right", fontsize=10)
    st.pyplot(fig)  

# Function to plot T-SNE with K-means clustering
def plot_tsne_with_kmeans(embeddings_tsne, kmeans_labels, all_lyrics, num_clusters):
    # Define a color map for clusters
    cluster_colors = plt.cm.get_cmap("tab10", num_clusters)

    fig = plt.figure(figsize=(36, 24))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster with a different color
    for i in range(num_clusters):
        ax.scatter(embeddings_tsne[kmeans_labels == i, 0], embeddings_tsne[kmeans_labels == i, 1], embeddings_tsne[kmeans_labels == i, 2],
                   c=[cluster_colors(i)] * len(embeddings_tsne[kmeans_labels == i]), label=f'Cluster {i+1}')

    # Annotate each point with its corresponding text (lyrics)
    for i, text in enumerate(all_lyrics):
        ax.text(embeddings_tsne[i, 0], embeddings_tsne[i, 1], embeddings_tsne[i, 2], text, fontsize=8)

    ax.set_title(f"T-SNE Visualization with K-means Clustering (n_clusters={num_clusters})")
    ax.legend()
    st.pyplot(fig) 

#Function for custom clustering
def custom_clusters(num_clusters, embeddings_tsne, kmeans_labels, all_lyrics, all_songs):
    # Create empty lists to store the clusters
    clusters = {i: [] for i in range(num_clusters)}

    # Separate the data points into their respective clusters
    for i in range(len(embeddings_tsne)):
        cluster_idx = kmeans_labels[i]
        clusters[cluster_idx].append((embeddings_tsne[i], all_lyrics[i], all_songs[i]))  # Include the song name

    # Display the lyrics and use OpenAI API to generate genre description
    for cluster_idx in range(num_clusters):
        with st.spinner(f'Now working on Cluster {cluster_idx}'):
            cluster_lyrics = [point[1] for point in clusters[cluster_idx]]
            cluster_songs = [point[2] for point in clusters[cluster_idx]] 

            st.write(f"Cluster {cluster_idx + 1} Lyrics:")
            for lyric, song in zip(cluster_lyrics, cluster_songs):
                # Split the lyric into lines and format each line
                formatted_lyric = "\n".join(lyric.split("\n"))
                st.write(f"Song: {song}")
                st.write(formatted_lyric)
                st.write("\n" + "="*50 + "\n") 

            # Create cluster_text from the cluster lyrics
            cluster_text = " ".join(cluster_lyrics) 

            # OpenAI API call to generate a genre description for the cluster
            prompt = """
            You will be given a cluster of lyrics. Based on the content of the lyrics, please provide a one word or phrase description of the content. You will 
            be making a custom genre that describes a user's listening tendencies. 
            Here is the prompt:  
            """
        
            input_text = prompt + cluster_text

            # Initialize OpenAI client
            api_key = st.secrets["OPENAI_KEY"]
            client = OpenAI(api_key=api_key)  
            
            # Make the API call
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[
                    {
                        "role": "user",
                        "content": input_text,
                    },
                ],
            )

        # Extract and display the genre description generated by GPT
        genre_description = completion.choices[0].message.content
        st.write(f"Cluster {cluster_idx + 1} Genre Description: {genre_description}")
        st.write("*"*50)  

run()