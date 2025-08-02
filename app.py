from flask import Flask, render_template, request, jsonify, url_for, session
from flask_cors import CORS
import secrets
import pickle
from scipy.sparse import csr_matrix
import numpy as np

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# Load trained ALS model and metadata
try:
    with open("model/steam_als_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    game2id = model_data["game2id"]
    id2game = model_data["id2game"]
    metadata = model_data["metadata"]
    print(f"✅ Model loaded successfully with {len(metadata)} games")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/quiz", methods=["GET"])
def quiz():
    return render_template("quiz.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or not all(k in data for k in ["q1", "q2", "q3", "q4", "q5", "q6"]):
        return jsonify({"error": "Incomplete quiz data"}), 400

    try:
        tags = quiz_to_tags(data)
        preferred_platform = get_preferred_platform(data["q6"])

        print(f"Generated tags: {tags}")
        print(f"Preferred platform: {preferred_platform}")

        input_app_ids = get_similar_games_from_tags(tags)
        print(f"Similar games found: {input_app_ids}")

        if not input_app_ids:
            # Fallback: get some popular games based on tags only
            print("No matching games found, using fallback approach")
            result_games = get_fallback_games(tags, preferred_platform)
            session["result_game"] = result_games
            return jsonify({"redirect": url_for("show_result")})

        recommended = recommend_games_for_app_ids(input_app_ids, top_n=15)
        print(f"Recommended games: {recommended}")

        # If no recommendations or recommendations failed, use fallback
        if not recommended:
            print("Using fallback games due to recommendation failure")
            result_games = get_fallback_games(tags, preferred_platform)
            session["result_game"] = result_games
            return jsonify({"redirect": url_for("show_result")})

        # Get game metadata for each recommendation and filter by platform
        result_games = []
        for app_id in recommended:
            if app_id not in metadata:
                continue

            game = metadata[app_id]

            # Check platform compatibility
            if preferred_platform != "all" and not is_platform_compatible(
                game, preferred_platform
            ):
                continue

            result_games.append(
                {
                    "title": game.get("title", "Unknown Game"),
                    "description": game.get("description", "No description available"),
                    "tags": game.get("tags", "No tags"),
                    "price": float(game.get("price_final", 0)),  # Already in dollars
                    "platforms": {
                        "Windows": game.get("win", False),
                        "Mac": game.get("mac", False),
                        "Linux": game.get("linux", False),
                        "Steam Deck": game.get("steam_deck", False),
                    },
                }
            )

            # Stop when we have 5 games that match the platform preference
            if len(result_games) >= 5:
                break

        # If we don't have enough games with platform filter, add more without filter
        if len(result_games) < 5:
            for app_id in recommended:
                if len(result_games) >= 5:
                    break

                if app_id not in metadata:
                    continue

                # Skip if already added
                if any(
                    g["title"] == metadata[app_id].get("title", "")
                    for g in result_games
                ):
                    continue

                game = metadata[app_id]
                result_games.append(
                    {
                        "title": game.get("title", "Unknown Game"),
                        "description": game.get(
                            "description", "No description available"
                        ),
                        "tags": game.get("tags", "No tags"),
                        "price": float(game.get("price_final", 0)),
                        "platforms": {
                            "Windows": game.get("win", False),
                            "Mac": game.get("mac", False),
                            "Linux": game.get("linux", False),
                            "Steam Deck": game.get("steam_deck", False),
                        },
                    }
                )

        session["result_game"] = result_games
        return jsonify({"redirect": url_for("show_result")})

    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/result", methods=["GET"])
def show_result():
    game = session.get("result_game", [])
    return render_template("result_new.html", result=game)


# -------- Quiz Mapping Functions -------- #


def quiz_to_tags(answers):
    tags = set()

    # Question 1: Social gatherings (multiplayer vs singleplayer preference)
    if int(answers["q1"]) >= 4:
        tags.update(["multiplayer", "co-op", "fps", "online"])
    else:
        tags.update(["singleplayer", "indie", "casual"])

    # Question 2: Travel and exploration
    if int(answers["q2"]) >= 4:
        tags.update(["exploration", "open world", "adventure"])

    # Question 3: Trust others (story-rich games)
    if int(answers["q3"]) >= 4:
        tags.update(["story rich", "visual novel", "rpg", "narrative"])

    # Question 4: Movie genre preference
    movie_map = {
        1: ["horror", "survival", "psychological horror"],
        2: ["romance", "visual novel", "dating sim"],
        3: ["sci-fi", "strategy", "space"],
        4: ["mystery", "puzzle", "detective"],
        5: ["action", "shooter", "combat"],
        6: ["comedy", "casual", "funny"],
    }
    tags.update(movie_map.get(int(answers["q4"]), []))

    # Question 5: Music genre preference
    music_map = {
        1: ["relaxing", "atmospheric", "meditative"],
        2: ["rhythm", "action", "fast-paced"],
        3: ["funky", "arcade", "retro"],
        4: ["indie", "jazz", "artistic"],
        5: ["anime", "visual novel", "japanese"],
    }
    tags.update(music_map.get(int(answers["q5"]), []))

    return list(tags)


def get_preferred_platform(q6_answer):
    """Convert Q6 answer to platform preference"""
    platform_map = {1: "windows", 2: "mac", 3: "linux", 4: "steam_deck", 5: "all"}
    return platform_map.get(int(q6_answer), "all")


def is_platform_compatible(game_metadata, preferred_platform):
    """Check if game is compatible with preferred platform"""
    if preferred_platform == "all":
        return True

    platform_keys = {
        "windows": "win",
        "mac": "mac",
        "linux": "linux",
        "steam_deck": "steam_deck",
    }

    platform_key = platform_keys.get(preferred_platform)
    if platform_key:
        return game_metadata.get(platform_key, False)

    return True


def get_similar_games_from_tags(tags, top_n=5):
    """Find games that match the generated tags"""
    candidates = []

    for app_id, meta in metadata.items():
        if not isinstance(meta.get("tags", ""), str):
            continue

        game_tags = meta["tags"].lower()
        tag_match = 0

        # Count matching tags (more flexible matching)
        for tag in tags:
            if tag.lower() in game_tags:
                tag_match += 1

        if tag_match > 0:
            candidates.append((app_id, tag_match))

    # Sort by number of matching tags
    candidates = sorted(candidates, key=lambda x: -x[1])
    result = [app_id for app_id, _ in candidates[:top_n]]

    print(f"Found {len(candidates)} candidates, returning top {len(result)}")
    return result


def recommend_games_for_app_ids(app_ids, top_n=15):
    """Use ALS model to recommend games based on input games"""
    known_ids = []

    # Get internal IDs for the input app_ids
    for aid in app_ids:
        if aid in game2id:
            known_ids.append(game2id[aid])

    if not known_ids:
        print("No known game IDs found")
        return list(metadata.keys())[:top_n]  # Return some random games

    # Create user vector
    user_vector = np.zeros(len(game2id))
    for i in known_ids:
        user_vector[i] = 1.0

    user_csr = csr_matrix(user_vector)

    try:
        # Get recommendations from the model
        recommended_items = model.recommend(
            userid=0, user_items=user_csr, N=top_n, filter_items=known_ids
        )

        print(f"Model returned recommendations of type: {type(recommended_items)}")
        print(
            f"Recommendations shape/length: {getattr(recommended_items, 'shape', len(recommended_items)) if hasattr(recommended_items, '__len__') else 'unknown'}"
        )

        # Handle the recommended items properly
        result = []

        # The implicit library returns a tuple of (ids, scores)
        if isinstance(recommended_items, tuple) and len(recommended_items) == 2:
            item_ids, scores = recommended_items
            print(f"Got {len(item_ids)} item IDs")

            for game_id in item_ids:
                try:
                    game_id = int(game_id)
                    # Convert back to app_id
                    if game_id in id2game:
                        app_id = id2game[game_id]
                        if app_id and app_id in metadata:
                            result.append(app_id)
                except (ValueError, TypeError) as e:
                    print(f"Error converting game_id {game_id}: {e}")
                    continue
        else:
            # Fallback: try to process as individual items
            for item in recommended_items:
                try:
                    if isinstance(item, (int, np.integer)):
                        game_id = int(item)
                    elif hasattr(item, "item"):
                        game_id = int(item.item())
                    else:
                        continue

                    # Convert back to app_id
                    if game_id in id2game:
                        app_id = id2game[game_id]
                        if app_id and app_id in metadata:
                            result.append(app_id)

                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Error processing item {item}: {e}")
                    continue

        print(f"Processed {len(result)} valid recommendations")
        return result[:top_n]

    except Exception as e:
        print(f"Error in recommendation: {e}")
        # Fallback: return the input games and some random ones
        fallback_games = list(app_ids) + list(metadata.keys())[:top_n]
        return fallback_games[:top_n]


def get_fallback_games(tags, preferred_platform, top_n=5):
    """Get fallback games when model recommendations fail"""
    candidates = []

    for app_id, meta in metadata.items():
        if not isinstance(meta.get("tags", ""), str):
            continue

        # Check platform compatibility first
        if preferred_platform != "all" and not is_platform_compatible(
            meta, preferred_platform
        ):
            continue

        game_tags = meta["tags"].lower()
        tag_match = sum(1 for tag in tags if tag.lower() in game_tags)

        if tag_match > 0:
            candidates.append((app_id, tag_match))

    # Sort by number of matching tags
    candidates = sorted(candidates, key=lambda x: -x[1])

    # If no platform-specific matches, try without platform filter
    if len(candidates) < top_n and preferred_platform != "all":
        print(f"Not enough platform-specific games, expanding search")
        for app_id, meta in metadata.items():
            if app_id in [c[0] for c in candidates]:  # Skip already added
                continue

            if not isinstance(meta.get("tags", ""), str):
                continue

            game_tags = meta["tags"].lower()
            tag_match = sum(1 for tag in tags if tag.lower() in game_tags)

            if tag_match > 0:
                candidates.append((app_id, tag_match))

    # Get the games and format them
    result_games = []
    for app_id, _ in candidates[:top_n]:
        game = metadata[app_id]
        result_games.append(
            {
                "title": game.get("title", "Unknown Game"),
                "description": game.get("description", "No description available"),
                "tags": game.get("tags", "No tags"),
                "price": float(game.get("price_final", 0)),
                "platforms": {
                    "Windows": game.get("win", False),
                    "Mac": game.get("mac", False),
                    "Linux": game.get("linux", False),
                    "Steam Deck": game.get("steam_deck", False),
                },
            }
        )

    print(f"Fallback returned {len(result_games)} games")
    return result_games


if __name__ == "__main__":
    app.run(debug=True)
