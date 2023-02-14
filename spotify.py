import time
import spotipy
from flask import Flask, redirect, url_for, session
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import spotipy.util as util
import pandas as pd
import numpy as np
import random as rand
import string as string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from main import app
import requests
import logging


CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = 'your_secret_here


def create_spotify_oauth():
    return SpotifyOAuth(client_id=CLIENT_ID, 
                        client_secret=CLIENT_SECRET,
                        redirect_uri=url_for("authorize", _external=True),
                        scope="playlist-modify-public playlist-modify-private playlist-read-private playlist-read-collaborative user-top-read")

def get_token():
    token_valid = False
    token_info = session.get("token_info", {})

    if not(session.get("token_info", False)):
        token_valid = False
        return token_info, token_valid
    
    now = int(time.time())
    is_token_expired = session.get("token_info").get("expires_at") - now < 60

    if(is_token_expired):
        sp_oauth = create_spotify_oauth()
        token_info = sp_oauth.refresh_access_token(session.get("token_info").get("refresh_token"))

    token_valid = True
    return token_info, token_valid

def get_current_username():
    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')
    sp = spotipy.Spotify(auth=session.get("token_info").get("access_token"))
    username = sp.current_user()
    return username

def get_playlist(playlist_uri):
    sp = spotipy.Spotify(auth=session.get("token_info").get("access_token"))
    return sp.playlist(playlist_uri)

#Create dataframe from playlist
def create_playlist_df(playlist):
    track_name = []
    track_id = []
    artist = []
    album = []
    duration = []
    popularity = []
    for items in playlist["tracks"]["items"]:
        try:
            track_name.append(items["track"]["name"])
            track_id.append(items["track"]["id"])
            artist.append(items["track"]["artists"][0]["name"])
            duration.append(items["track"]["duration_ms"])
            album.append(items["track"]["album"]["name"])
            popularity.append(items["track"]["popularity"])
        except TypeError:
            pass
    df = pd.DataFrame({ "track_id": track_id,
                        "track_name": track_name,
                        "album": album,
                        "artist": artist,
                        "duration": duration,
                        "popularity": popularity})
    return df

#Create recommended dataframe from playlist
def create_recommended_df(playlist):
    track_name = []
    track_id = []
    artist = []
    album = []
    duration = []
    popularity = []
    for items in playlist["tracks"]:
        try:
            track_name.append(items["name"])
            track_id.append(items["id"])
            artist.append(items["artists"][0]["name"])
            duration.append(items["duration_ms"])
            album.append(items["album"]["name"])
            popularity.append(items["popularity"])
        except TypeError:
            pass
    df = pd.DataFrame({ "track_id": track_id,
                        "track_name": track_name,
                        "album": album,
                        "artist": artist,
                        "duration": duration,
                        "popularity": popularity})
    return df

#Append audio features to dataframe
def append_audio_features(df):
    sp = spotipy.Spotify(auth=session.get("token_info").get("access_token"))
    audio_features = sp.audio_features(df["track_id"][:])
    if None in audio_features:
        NA_idx = [i for i,v in enumerate(audio_features) if v == None]
        df.drop(NA_idx, inplace=True)
        for i in NA_idx:
            audio_features.pop(i)
    assert len(audio_features) == len(df["track_id"][:])
    feature_columns = list(audio_features[0].keys())[:-7]
    features_list = []
    for features in audio_features:
        try:
            song_features = [features[column] for column in feature_columns]
            features_list.append(song_features)
        except TypeError:
            pass
    df_features = pd.DataFrame(features_list, columns=feature_columns)
    df = pd.concat([df, df_features], axis=1)
    df = df.drop('mode', 1)
    df = df.drop('key', 1)
    return df

#Create recommendations playlist based on seeded tracks
def recommended_df(playlist_df):
    sp = spotipy.Spotify(auth=session.get("token_info").get("access_token"))
    seed_tracks = playlist_df["track_id"].tolist()
    recommended_dfs = []
    for i in range(5, len(seed_tracks) + 1, 5):
        recommendations = sp.recommendations(seed_tracks = seed_tracks[i-5:i], limit=25)
        recommended_df = append_audio_features(create_recommended_df(recommendations))
        recommended_dfs.append(recommended_df)
    recommended_df = pd.concat(recommended_dfs)
    recommended_df.reset_index(drop=True, inplace=True)
    return recommended_df

#Create similarity score between playlist and recommended
def create_similarity_score(df1, df2):
    features = df1.columns[6:]
    df_features1, df_features2 = df1[features], df2[features]
    scaler = MinMaxScaler()
    df_features_scaled1, df_features_scaled2 = scaler.fit_transform(df_features1), scaler.fit_transform(df_features2)
    cosine_sim = cosine_similarity(df_features_scaled1, df_features_scaled2)
    return cosine_sim

#Create final recommendations dataframe
def create_final_recommendation(playlist_df, recommended_df, similarity_score):
    final_recommendations_df = recommended_df.iloc[[np.argmax(i) for i in similarity_score]]
    final_recommendations_df = final_recommendations_df.drop_duplicates()
    final_recommendations_df = final_recommendations_df[~final_recommendations_df["track_name"].isin(playlist_df["track_name"])]
    final_recommendations_df.reset_index(drop=True, inplace=True)
    return final_recommendations_df

#Add dataframe to spotify playlist
def add_playlist_to_spotify(final_recommendation_df, playlist_name):
    sp = spotipy.Spotify(auth=session.get("token_info").get("access_token"))
    username = get_current_username()['id']
    final_recommendations_df = final_recommendation_df.sort_values(['popularity'], ascending=False)
    final_recommendations_df = final_recommendations_df.head(50)
    playlist_recs = sp.user_playlist_create(username, name=playlist_name, description="Playlist created using Python")
    rec_list = list(final_recommendations_df["track_id"])
    sp.user_playlist_add_tracks(username, playlist_recs["id"], ["spotify:track:" + track for track in rec_list])

#Align to left
def left_align(df):
    left_aligned_df = df.style.set_properties(**{"text-align" : "left"})
    left_aligned_df = left_aligned_df.set_table_styles([dict(selector="th", props=[("text-align", "left")])])
    return left_aligned_df

#Get top 50 tracks of user
def get_top_tracks():
    sp = spotipy.Spotify(auth=session.get("token_info").get("access_token"))
    results = sp.current_user_top_tracks(time_range="short_term", limit=50)
    top_tracks_df = pd.DataFrame(columns=['Rank', 'Name', 'Artist'])
    for i, item in enumerate(results['items']):
        temp_df = {'Rank': i+1, 'Name': item["name"], 'Artist': item["artists"][0]["name"]}
        # print(i, item["name"], "//", item["artists"][0]["name"])
        top_tracks_df = top_tracks_df.append(temp_df, ignore_index=True)
    top_tracks_df = top_tracks_df.set_index(['Rank', 'Name', 'Artist'])
    return top_tracks_df
