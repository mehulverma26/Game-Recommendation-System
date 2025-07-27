from flask import Flask, render_template, request, jsonify, url_for, session
from flask_cors import CORS
import secrets

app=Flask(__name__)
app.secret_key=secrets.token_hex(16)
CORS(app)

@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/quiz",methods=["GET"])
def quiz():
    return render_template("quiz.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    personality = data.get("personality")

    if not personality or not isinstance(personality, list):
        return jsonify({"error": "Invalid Input"}), 400

    game = get_recommended_game(personality)

    # Save result to session and tell JS to redirect
    session["result_game"] = game
    return jsonify({"redirect": url_for("show_result")})

@app.route("/result", methods=["GET"])
def show_result():
    game = session.get("result_game", "No result found.")
    return render_template("result_new.html", result=game)

'''
@app.route("/result", methods=["GET"])
def show_result():
    game = session.get("result_game", "No result found.")
    return render_template("result.html", result=game)
'''

def get_recommended_game(personality):
    return{
        1:"Minecraft",
        2:"The Witcher 3",
        3: "Animal Crossing",
        4: "Elden Ring"
    }.get(personality[0],"Tetris")

if __name__=="__main__":
    app.run(debug=True)