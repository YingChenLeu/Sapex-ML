from apscheduler.schedulers.background import BackgroundScheduler
import threading
# uvicorn utils.predictor:app --reload
from fastapi.middleware.cors import CORSMiddleware
import os 
from dotenv import load_dotenv  

load_dotenv()
import json, base64
certificate = json.loads(base64.b64decode(os.environ["FIREBASE_CERTIFICATE"]))

import firebase_admin
from firebase_admin import credentials

# Initialize Firebase app with credentials if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(certificate)
    firebase_admin.initialize_app(cred)

import random
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from firebase_admin import firestore

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = firestore.client()

def get_user_ocean(uid):
    doc_ref = db.collection('users').document(uid)
    doc = doc_ref.get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    traits = data.get('big_five_traits')
    if traits and isinstance(traits, list) and len(traits) == 5:
        return traits
    return None

def get_all_helpers(exclude_uid):
    helpers = []
    users_ref = db.collection('users')
    query = users_ref.where('online', '==', True)
    docs = query.stream()
    for doc in docs:
        if doc.id == exclude_uid:
            continue
        data = doc.to_dict()
        traits = data.get('big_five_traits')
        if traits and isinstance(traits, list) and len(traits) == 5:
            helpers.append((doc.id, traits))
    return helpers

# Simulate a model state (in-memory for demo purposes)
model_state = {}

def _feature_key(user1_traits, user2_traits):
    # Generate a hashable key for tracking interactions
    return tuple(user1_traits + user2_traits)

def predict_match(user1_traits, user2_traits):
    key = _feature_key(user1_traits, user2_traits)
    if key in model_state:
        return model_state[key]['score']
    else:
        # Return a random guess if no prior data
        return round(random.uniform(0.4, 0.6), 2)

def update_model(user1_traits, user2_traits, feedback_score):
    key = _feature_key(user1_traits, user2_traits)
    model_state[key] = {
        "score": round(feedback_score / 10.0, 2)  # Normalize feedback to 0–1
    }

feedback_docs = db.collection("esupport").stream()
feedback_pairs = []
for doc in feedback_docs:
    data = doc.to_dict()
    seeker_uid = data.get("seeker_uid")
    helper_uid = data.get("helper_uid")
    actual_score = data.get("actual")
    if seeker_uid and helper_uid and isinstance(actual_score, (int, float)):
        seeker = get_user_ocean(seeker_uid)
        helper = get_user_ocean(helper_uid)
        if seeker and helper:
            feedback_pairs.append((seeker, helper, actual_score))


import firebase_admin
from firebase_admin import credentials, firestore


import random
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse




def get_user_ocean(uid):
    # Query the user document from Firebase
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    if not user_doc.exists:
        return None
    data = user_doc.to_dict()
    personality = data.get("bigFivePersonality", {})
    traits = [
        personality.get("Extraversion", 0),
        personality.get("Agreeableness", 0),
        personality.get("Conscientiousness", 0),
        personality.get("Neuroticism", 0),
        personality.get("Openness", 0)
    ]
    if all(isinstance(val, (int, float)) for val in traits):
        return traits
    else:
        return None

def get_all_helpers(exclude_uid):
    helpers = []
    users_ref = db.collection("users")
    docs = users_ref.stream()

    for doc in docs:
        data = doc.to_dict()
        uid = doc.id

        # Skip if it's the same as the seeker, offline, busy, or isAdmin
        if uid == exclude_uid or not data.get("online") or data.get("busy") or data.get("isAdmin"):
            continue

        personality = data.get("bigFivePersonality", {})
        traits = [
            personality.get("Extraversion", 0),
            personality.get("Agreeableness", 0),
            personality.get("Conscientiousness", 0),
            personality.get("Neuroticism", 0),
            personality.get("Openness", 0)
        ]

        if all(isinstance(val, (int, float)) for val in traits):
            helpers.append((uid, traits))

    return helpers



@app.get("/match")
def match(
    uid: str = Query(..., description="UID of the person seeking help"),
    problem_type: str = Query(None, description="Type of emotional issue")
):
    seeker_traits = get_user_ocean(uid)

    if not seeker_traits:
        return JSONResponse(status_code=404, content={"error": "User not found or missing traits"})

    helpers = get_all_helpers(uid)
    if not helpers:
        return JSONResponse(status_code=404, content={"error": "No available helpers"})

    best_score = -1
    selected_uid = None
    selected_traits = None
    for helper_uid, traits in helpers:
        curr_score = predict_match(seeker_traits, traits)
        if curr_score > best_score:
            best_score = curr_score
            selected_uid = helper_uid
            selected_traits = traits
    score = best_score

    return {
        "seeker_uid": uid,
        "helper_uid": selected_uid,
        "predicted_score": score,
        "problem_type": problem_type
    }


# --- Cold-start scoring: default weights per problem type ---
# [Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness]
DEFAULT_WEIGHTS_BY_PROBLEM = {
    "friendship":  [0.3, 0.4, 0.0, 0.1, 0.1,  0.5, 0.4, 0.0, 0.1, 0.1],
    "loneliness":  [0.2, 0.4, 0.0, 0.2, 0.1,  0.4, 0.5, 0.0, 0.2, 0.1],
    "heartbreak":  [0.1, 0.3, 0.1, 0.0, 0.1,  0.2, 0.7, 0.1, -0.4, 0.3],
    "burnout":     [0.0, 0.1, 0.3, -0.2, 0.1, 0.0, 0.3, 0.6, -0.4, 0.1],
    "stress":      [0.0, 0.2, 0.2, -0.3, 0.1, 0.0, 0.6, 0.2, -0.5, 0.3],
    "guidance":    [0.3, 0.3, 0.2, 0.0, 0.1,  0.1, 0.3, 0.4, 0.0, 0.3],
    "study":       [0.2, 0.2, 0.4, -0.1, 0.1, 0.1, 0.2, 0.6, -0.1, 0.1],
    "default":     [0.2, 0.2, 0.2, 0.2, 0.2,  0.2, 0.2, 0.2, 0.2, 0.2],
}


# --- Cold-start match endpoint ---
@app.get("/coldstart_match")
def coldstart_match(
    uid: str = Query(..., description="ID of person seeking help"),
    problem_type: str = Query("default", description="Type of emotional issue")
):
    """
    Cold-start compatibility scoring using default weights (no ML model).
    - Fetches seeker and helper traits
    - Uses default weights per problem type
    - Calculates compatibility via a simple weighted-difference score
    - Returns relevant info and strategy explanation
    """
    seeker = get_user_ocean(uid)
    helpers = get_all_helpers(uid)

    if not seeker:
        return JSONResponse(status_code=404, content={"error": "Seeker not found."})

    if not helpers:
        return JSONResponse(status_code=404, content={"error": "No available helpers."})

    weights = get_problem_weights(problem_type)

    def score_match(seeker_traits, helper_traits, weights):
        seeker_weights = weights[:5]
        helper_weights = weights[5:]
        score = sum(sw * s + hw * h for s, h, sw, hw in zip(seeker_traits, helper_traits, seeker_weights, helper_weights))
        return round(score, 2)

    best_score = -1
    best_match = None

    for helper_uid, traits in helpers:
        score = score_match(seeker, traits, weights)
        if score > best_score:
            best_score = score
            best_match = (helper_uid, traits)
            
    return {
        "strategy": "coldstart-default-weights",
        "seeker_id": uid,
        "helper_uid": best_match[0],
        "predicted_score": best_score,
        "problem_type": problem_type,
        "seeker_traits": seeker,
        "helper_traits": best_match[1]
    }

# --- Genetic Algorithm endpoint for evolving weights ---
from deap import base, creator, tools
import pickle

import os

evolved_weights_by_problem = {}

def load_weights():
    global evolved_weights_by_problem
    # Try Firestore first
    try:
        doc = db.collection("model_meta").document("evolved_weights").get()
        if doc.exists:
            data = doc.to_dict() or {}
            by_problem = data.get("by_problem") or {}
            if isinstance(by_problem, dict):
                # Validate shape: only keep length-10 lists/tuples
                evolved_weights_by_problem = {
                    k: list(v)
                    for k, v in by_problem.items()
                    if isinstance(v, (list, tuple)) and len(v) == 10
                }
                return
    except Exception as e:
        print(f"load_weights Firestore error: {e}")

    # Legacy fallback: local pickle (ephemeral on Render)
    try:
        if os.path.exists("evolved_weights.pkl"):
            with open("evolved_weights.pkl", "rb") as f:
                evolved_weights_by_problem = pickle.load(f)
            # Best-effort: persist legacy weights to Firestore for durability
            try:
                db.collection("model_meta").document("evolved_weights").set({
                    "by_problem": evolved_weights_by_problem
                }, merge=True)
            except Exception as e2:
                print(f"migrate legacy weights error: {e2}")
    except Exception as e:
        print(f"load_weights pickle error: {e}")

# --- Helper: resolve weights for a given problem type, preferring trained weights ---
def get_problem_weights(problem_type: str):
    """
    Return a length-10 weight vector for the given problem type, preferring
    GA-trained weights if available, otherwise falling back to defaults.
    Order: first 5 for seeker traits, last 5 for helper traits.
    """
    # Try GA-evolved weights for the requested type
    w = evolved_weights_by_problem.get(problem_type)
    if isinstance(w, (list, tuple)) and len(w) == 10:
        return list(w)

    # Fall back to GA-evolved default, if present
    w = evolved_weights_by_problem.get("default")
    if isinstance(w, (list, tuple)) and len(w) == 10:
        return list(w)

    # Finally, use hardcoded defaults
    return DEFAULT_WEIGHTS_BY_PROBLEM.get(
        problem_type,
        DEFAULT_WEIGHTS_BY_PROBLEM["default"]
    )

def save_weights():
    # Primary: save to Firestore so weights persist across restarts/redeploys
    try:
        db.collection("model_meta").document("evolved_weights").set({
            "by_problem": evolved_weights_by_problem
        }, merge=True)
    except Exception as e:
        print(f"save_weights Firestore error: {e}")

    # Optional: also write local pickle as a warm-cache (ephemeral on Render)
    try:
        with open("evolved_weights.pkl", "wb") as f:
            pickle.dump(evolved_weights_by_problem, f)
    except Exception as e:
        print(f"save_weights pickle error: {e}")

load_weights()

@app.get("/evolve_weights")
def evolve_weights(problem_type: str = Query("default", description="Problem type to evolve weights for")):
    NUM_TRAITS = 10

    # Only create once (avoid re-creation error)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=NUM_TRAITS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Query real seekers from Firestore
    seeker_docs = db.collection("users").where("seekingHelp", "==", True).stream()
    seekers = []
    for doc in seeker_docs:
        personality = doc.to_dict().get("bigFivePersonality", {})
        traits = [
            personality.get("Extraversion", 0),
            personality.get("Agreeableness", 0),
            personality.get("Conscientiousness", 0),
            personality.get("Neuroticism", 0),
            personality.get("Openness", 0)
        ]
        if all(isinstance(val, (int, float)) for val in traits):
            seekers.append(traits)
    helpers = [traits for _, traits in get_all_helpers("")]

    seeker_helper_pairs = [(s, h) for s in seekers for h in helpers]

    def fitness_function(weights):
        seeker_weights = weights[:5]
        helper_weights = weights[5:]
        scores = []
        for seeker, helper in seeker_helper_pairs:
            score = sum(sw * s + hw * h for s, h, sw, hw in zip(seeker, helper, seeker_weights, helper_weights))
            scores.append(score)
        return (sum(scores) / len(scores),)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    NGEN = 30

    for gen in range(NGEN):
        offspring = toolbox.select(pop, len(pop) - 1)
        offspring = list(map(toolbox.clone, offspring))
        elite = tools.selBest(pop, k=1)[0]
        offspring.append(elite)

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

    best = tools.selBest(pop, 1)[0]
    evolved_weights_by_problem[problem_type] = list(best)
    save_weights()
    return {"problem_type": problem_type, "best_weights": list(best)}


# --- Genetic Algorithm endpoint for training weights with synthetic data ---
@app.post("/train_ga_weights")
def train_ga_weights(problem_type: str = Query("default", description="Problem type to train weights for")):
    """
    Train GA weights for emotional support matching using synthetic seeker/helper data.
    """
    from deap import base, creator, tools

    NUM_TRAITS = 10

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=NUM_TRAITS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    feedback_docs = db.collection("esupport").stream()
    feedback_pairs = []
    for doc in feedback_docs:
        data = doc.to_dict()
        seeker_uid = data.get("seeker_uid")
        helper_uid = data.get("helper_uid")
        actual_score = data.get("actual")
        if (
            seeker_uid and helper_uid and
            isinstance(actual_score, (int, float)) and
            data.get("type") == problem_type
        ):
            seeker = get_user_ocean(seeker_uid)
            helper = get_user_ocean(helper_uid)
            if seeker and helper:
                feedback_pairs.append((seeker, helper, actual_score))

    def fitness_function(weights):
        seeker_weights = weights[:5]
        helper_weights = weights[5:]
        scores = []
        for seeker, helper, actual_score in feedback_pairs:
            predicted_score = sum(sw * s + hw * h for s, h, sw, hw in zip(seeker, helper, seeker_weights, helper_weights))
            error = abs(predicted_score - actual_score)
            scores.append(1 - error)
        return (sum(scores) / len(scores),)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=60)
    generations = 40

    for gen in range(generations):
        offspring = toolbox.select(pop, len(pop) - 1)
        offspring = list(map(toolbox.clone, offspring))
        elite = tools.selBest(pop, k=1)[0]
        offspring.append(elite)

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

    best = tools.selBest(pop, 1)[0]
    evolved_weights_by_problem[problem_type] = list(best)
    save_weights()
    return {"best_evolved_weights": list(best)}


# --- APScheduler: weekly training job ---
# Make sure train_ga_weights is defined before scheduler setup

scheduler = BackgroundScheduler()

def weekly_train_job():
    try:
        for ptype in [
            "default",
            "friendship",
            "loneliness",
            "heartbreak",
            "burnout",
            "stress",
            "guidance",
            "study",
        ]:
            train_ga_weights(ptype)
    except Exception as e:
        print(f"Training failed: {e}")

# Schedule the job: every Saturday at 3:00 AM server time
scheduler.add_job(weekly_train_job, trigger='cron', day_of_week='sat', hour=3, minute=0)
scheduler.start()

# Prevent scheduler from being garbage collected
threading.Thread(target=scheduler.start, daemon=True).start()