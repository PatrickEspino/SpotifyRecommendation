# from crypt import methods
from typing import final
from flask import Flask, redirect, render_template, request, url_for, session, make_response
import pandas as pd
import spotify

app = Flask(__name__)
app.secret_key = "random_secret_key"
app.config['SESSION_COOKIE_NAME'] = "cookie"

@app.route("/")
def default():
    sp_oauth = spotify.create_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/authorize")
def authorize():
    sp_oauth = spotify.create_spotify_oauth()
    session.clear()
    code = request.args.get("code")
    token_info = sp_oauth.get_access_token(code)
    session["token_info"] = token_info
    return redirect("/home")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/recommend", methods=["POST", "GET"])
def recommend():
    if(request.method == 'POST'):
        profile = spotify.get_current_username()
        print(profile)
        URL = request.form["URL_name"]
        playlist_uri = URL
        playlist = spotify.get_playlist(playlist_uri)
        playlist_df = spotify.create_playlist_df(playlist)
        playlist_df = spotify.append_audio_features(playlist_df)

        recommended_df = spotify.recommended_df(playlist_df)
        similarity_score = spotify.create_similarity_score(playlist_df, recommended_df)
        final_recommended_df = spotify.create_final_recommendation(playlist_df, recommended_df, similarity_score)
        # output_df = final_recommended_df[['track_name', 'artist']]
        # output_df = spotify.left_align(output_df)
        playlist_name = request.form["playlist_name"]
        return render_template('results.html', tables=[final_recommended_df.to_html(classes='data')], titles=final_recommended_df.columns, add_playlist=spotify.add_playlist_to_spotify(final_recommended_df, playlist_name))
    return render_template("recommend.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/top_tracks")
def top_tracks():
    top_tracks_df = spotify.get_top_tracks()
    # return top_tracks_df
    return render_template("toptracks.html", tables=[top_tracks_df.to_html(classes="data")], titles=top_tracks_df.columns)

if __name__ == "__main__":
    app.run(debug=True)