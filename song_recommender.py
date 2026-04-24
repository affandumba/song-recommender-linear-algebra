"""
╔══════════════════════════════════════════════════════════════╗
║   Song Recommendation System — Linear Algebra Pipeline       ║
║   UE24MA241B | PES University                                ║
║   Users: Affan, Atharv, Rizwan, Mohit                       ║
╚══════════════════════════════════════════════════════════════╝

Core Concepts Used:
  - Matrix Representation
  - Vectors in R^6
  - Norms        : ||v|| = sqrt(sum of squares)
  - Inner Product: <u,v> = dot product
  - Cosine Similarity: cos(θ) = <u,v> / (||u|| * ||v||)
  - Orthogonal Projection: predict unseen ratings
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# SONG DATABASE
# ──────────────────────────────────────────────────────────────
# 6 songs users RATE (1 per genre) — these form the rating matrix
RATING_SONGS = {
    "Pop"     : "Blinding Lights – The Weeknd",
    "Hip-Hop" : "HUMBLE. – Kendrick Lamar",
    "Rock"    : "Bohemian Rhapsody – Queen",
    "R&B"     : "No One – Alicia Keys",
    "EDM"     : "Levels – Avicii",
    "Indie"   : "Do I Wanna Know? – Arctic Monkeys",
}

# Genres in fixed order (important for vector consistency)
GENRES = ["Pop", "Hip-Hop", "Rock", "R&B", "EDM", "Indie"]

# 18 unseen catalogue songs (3 per genre) — system recommends from these
CATALOGUE = {
    "Pop"     : ["As It Was – Harry Styles",       "Anti-Hero – Taylor Swift",       "Stay – The Kid LAROI"],
    "Hip-Hop" : ["God's Plan – Drake",             "Sicko Mode – Travis Scott",      "Rockstar – Post Malone"],
    "Rock"    : ["Mr. Brightside – The Killers",    "Smells Like Teen Spirit – Nirvana","Thunder – Imagine Dragons"],
    "R&B"     : ["Golden – Joji",                   "Redbone – Childish Gambino",     "Save Your Tears – The Weeknd"],
    "EDM"     : ["Titanium – David Guetta",         "Lean On – Major Lazer",          "Animals – Martin Garrix"],
    "Indie"   : ["Fluorescent Adolescent – Arctic Monkeys", "505 – Arctic Monkeys",   "Sweater Weather – The Neighbourhood"],
}

USERS = ["Affan", "Atharv", "Rizwan", "Mohit"]

# ──────────────────────────────────────────────────────────────
# HELPER: Print separator
# ──────────────────────────────────────────────────────────────
def sep(char="=", n=62):
    print(char * n)

def section(title):
    print()
    sep()
    print(f"  {title}")
    sep()

# ──────────────────────────────────────────────────────────────
# STEP 1: COLLECT RATINGS FROM EACH USER
# ──────────────────────────────────────────────────────────────
def get_ratings():
    """Ask each user to rate 6 songs (1 per genre). Returns rating matrix A."""
    section("STEP 1 : RATING INPUT")
    print("  Each user will rate 6 songs — one from each genre.")
    print("  Enter a rating from 1 (hate it) to 5 (love it).\n")

    rating_matrix = []   # will become our matrix A (4 users x 6 genres)

    for user in USERS:
        print(f"\n  {'─'*50}")
        print(f"  🎵  Hello {user}! Please rate these 6 songs:")
        print(f"  {'─'*50}")
        user_ratings = []

        for genre in GENRES:
            song = RATING_SONGS[genre]
            while True:
                try:
                    val = int(input(f"  [{genre:10}]  {song:40} → "))
                    if 1 <= val <= 5:
                        user_ratings.append(float(val))
                        break
                    else:
                        print("    ⚠  Please enter a number between 1 and 5.")
                except ValueError:
                    print("    ⚠  Invalid input. Enter a number (1–5).")

        rating_matrix.append(user_ratings)
        print(f"  ✅  Thanks {user}! Your ratings have been recorded.")

    return np.array(rating_matrix)   # shape: (4, 6)

# ──────────────────────────────────────────────────────────────
# STEP 2: DISPLAY RATING MATRIX
# ──────────────────────────────────────────────────────────────
def display_matrix(A):
    section("STEP 2 : USER–GENRE RATING MATRIX  (Matrix A)")
    print(f"  {'User':10}", end="")
    for g in GENRES:
        print(f"  {g:12}", end="")
    print()
    sep("─")
    for i, user in enumerate(USERS):
        print(f"  {user:10}", end="")
        for val in A[i]:
            print(f"  {val:12.1f}", end="")
        print()

# ──────────────────────────────────────────────────────────────
# STEP 3: NORMS
# ──────────────────────────────────────────────────────────────
def compute_norms(A):
    section("STEP 3 : NORMS  ||v|| = sqrt( Σ vᵢ² )")
    print("  Measures the overall rating intensity of each user.\n")
    norms = np.linalg.norm(A, axis=1)
    for i, user in enumerate(USERS):
        vec = A[i]
        breakdown = " + ".join([f"{v:.0f}²" for v in vec])
        print(f"  ||{user:6}|| = sqrt({breakdown}) = {norms[i]:.4f}")
    return norms

# ──────────────────────────────────────────────────────────────
# STEP 4: INNER PRODUCTS
# ──────────────────────────────────────────────────────────────
def compute_inner_products(A):
    section("STEP 4 : INNER PRODUCTS  <u,v> = Σ uᵢ × vᵢ")
    print("  Measures raw similarity between users.\n")
    IP = A @ A.T   # (4x6) @ (6x4) = (4x4) inner product matrix

    # Display table
    print(f"  {'':10}", end="")
    for u in USERS:
        print(f"  {u:8}", end="")
    print()
    sep("─")
    for i, u in enumerate(USERS):
        print(f"  {u:10}", end="")
        for j in range(len(USERS)):
            print(f"  {IP[i,j]:8.1f}", end="")
        print()
    return IP

# ──────────────────────────────────────────────────────────────
# STEP 5: COSINE SIMILARITY + COMPATIBILITY SCORES
# ──────────────────────────────────────────────────────────────
def compute_cosine_similarity(A, IP, norms):
    section("STEP 5 : COSINE SIMILARITY  cos(θ) = <u,v> / (||u|| × ||v||)")
    print("  Normalised similarity — removes rating intensity bias.\n")

    n = len(USERS)
    cos_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            denom = norms[i] * norms[j]
            cos_sim[i, j] = IP[i, j] / denom if denom != 0 else 0

    # Display similarity matrix
    print(f"  {'':10}", end="")
    for u in USERS:
        print(f"  {u:10}", end="")
    print()
    sep("─")
    for i, u in enumerate(USERS):
        print(f"  {u:10}", end="")
        for j in range(n):
            print(f"  {cos_sim[i,j]:10.4f}", end="")
        print()

    # ── Pairwise Compatibility Scores ──
    print(f"\n  {'─'*56}")
    print("  🤝  FRIEND COMPATIBILITY SCORES")
    print(f"  {'─'*56}")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            score = cos_sim[i, j] * 100
            pairs.append((USERS[i], USERS[j], score))

    # Sort by score descending
    pairs.sort(key=lambda x: x[2], reverse=True)

    for u1, u2, score in pairs:
        bar_len = int(score / 5)          # scale to 20 chars
        bar = "█" * bar_len + "░" * (20 - bar_len)
        label = "🔥 Best match!" if score >= 85 else ("👍 Good match" if score >= 70 else "🔀 Different tastes")
        print(f"  {u1:8} ↔ {u2:8}  [{bar}]  {score:5.1f}%  {label}")

    return cos_sim

# ──────────────────────────────────────────────────────────────
# STEP 6: PREDICT CATALOGUE RATINGS VIA PROJECTION
# ──────────────────────────────────────────────────────────────
def predict_catalogue_ratings(A, cos_sim):
    """
    For each unseen catalogue song, predict how each user would rate it
    using weighted projection onto the nearest neighbour's genre vector.

    Method:
      - The catalogue song belongs to a genre (e.g. Action)
      - Each user's rating for that genre = A[user, genre_index]
      - Predicted rating for unseen song = weighted avg of top-2
        neighbours' genre ratings, weighted by cosine similarity
    """
    n_users = len(USERS)
    # predicted_ratings[user][song_name] = predicted score
    predicted_ratings = {u: {} for u in USERS}

    for genre_idx, genre in enumerate(GENRES):
        for song in CATALOGUE[genre]:
            for i, user in enumerate(USERS):
                # Find top-2 most similar neighbours (excluding self)
                sims = cos_sim[i].copy()
                sims[i] = -1
                top2 = np.argsort(sims)[::-1][:2]

                w_sum, w_total = 0.0, 0.0
                for n in top2:
                    w = sims[n]
                    neighbour_genre_rating = A[n, genre_idx]
                    w_sum   += w * neighbour_genre_rating
                    w_total += w

                if w_total > 0:
                    predicted = w_sum / w_total
                else:
                    # Fallback: use own genre rating
                    predicted = A[i, genre_idx]

                predicted_ratings[user][song] = round(predicted, 2)

    return predicted_ratings

# ──────────────────────────────────────────────────────────────
# STEP 7: INDIVIDUAL RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────
def individual_recommendations(A, predicted_ratings):
    section("STEP 7 : INDIVIDUAL RECOMMENDATIONS")
    print("  For each user → find their highest-rated genre →")
    print("  recommend the most similar unseen song from that genre.\n")

    user_top_songs = {}

    for i, user in enumerate(USERS):
        # Find the genre the user rated highest
        top_genre_idx = int(np.argmax(A[i]))
        top_genre     = GENRES[top_genre_idx]
        top_rating    = A[i, top_genre_idx]
        rated_song    = RATING_SONGS[top_genre]

        # Among catalogue songs of that genre, pick highest predicted
        candidates = CATALOGUE[top_genre]
        best_song   = max(candidates, key=lambda s: predicted_ratings[user][s])
        best_score  = predicted_ratings[user][best_song]

        user_top_songs[user] = best_song

        print(f"  {'─'*54}")
        print(f"  👤  {user}")
        print(f"      Loved genre    : {top_genre} (rated '{rated_song}' → {top_rating:.0f}/5)")
        print(f"      Recommended    : 🎵 {best_song}")
        print(f"      Predicted score: {best_score:.2f} / 5.00")

    print(f"  {'─'*54}")
    return user_top_songs

# ──────────────────────────────────────────────────────────────
# STEP 8: GROUP BLEND RECOMMENDATION
# ──────────────────────────────────────────────────────────────
def group_blend(predicted_ratings):
    section("STEP 8 : GROUP BLEND 🎶")
    print("  Method: Pick the catalogue song with the HIGHEST")
    print("  average predicted rating across all 4 friends.\n")

    # Collect all catalogue songs
    all_songs = [s for genre in GENRES for s in CATALOGUE[genre]]

    # Compute average predicted rating for each song across all users
    avg_scores = {}
    for song in all_songs:
        scores = [predicted_ratings[user][song] for user in USERS]
        avg_scores[song] = (np.mean(scores), scores)

    # Sort by average score
    ranked = sorted(avg_scores.items(), key=lambda x: x[1][0], reverse=True)

    # Show top 5
    print(f"  {'Rank':<5} {'Song':<42} {'Avg':>5}   {'Individual Scores'}")
    sep("─")
    for rank, (song, (avg, scores)) in enumerate(ranked[:5], 1):
        score_str = "  ".join([f"{USERS[i]}:{scores[i]:.1f}" for i in range(len(USERS))])
        marker = "  ⭐ GROUP PICK!" if rank == 1 else ""
        print(f"  #{rank:<4} {song:<42} {avg:>5.2f}   {score_str}{marker}")

    sep("─")
    best_song, (best_avg, best_scores) = ranked[0]
    print(f"\n  🎵  Best song for the group: \"{best_song}\"")
    print(f"      Average predicted rating : {best_avg:.2f} / 5.00")
    print(f"\n      Individual predicted ratings:")
    for i, user in enumerate(USERS):
        bar = "★" * round(best_scores[i]) + "☆" * (5 - round(best_scores[i]))
        print(f"        {user:8} → {bar}  ({best_scores[i]:.2f}/5)")

# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────
def main():
    print()
    sep("═")
    print("  🎵  SONG RECOMMENDATION SYSTEM")
    print("  📐  Powered by Linear Algebra (Norms & Inner Products)")
    print("  👥  Users: Affan | Atharv | Rizwan | Mohit")
    sep("═")

    # Step 1: Collect ratings
    A = get_ratings()

    # Step 2: Display matrix
    display_matrix(A)

    # Step 3: Norms
    norms = compute_norms(A)

    # Step 4: Inner products
    IP = compute_inner_products(A)

    # Step 5: Cosine similarity + compatibility
    cos_sim = compute_cosine_similarity(A, IP, norms)

    # Step 6: Predict catalogue ratings via projection
    section("STEP 6 : ORTHOGONAL PROJECTION → PREDICTING UNSEEN RATINGS")
    print("  Formula: predicted = Σ(sim × neighbour_genre_rating) / Σ(sim)")
    print("  Projecting each user's taste vector onto neighbour subspace...\n")
    predicted_ratings = predict_catalogue_ratings(A, cos_sim)
    print("  ✅  Predictions complete for all 18 catalogue songs × 4 users.")

    # Step 7: Individual recommendations
    individual_recommendations(A, predicted_ratings)

    # Step 8: Group blend
    group_blend(predicted_ratings)

    # Final summary
    section("PIPELINE COMPLETE ✅")
    print("  Real-World Data  →  Matrix A  →  User Vectors in R⁶")
    print("  →  Norms (||v||)  →  Inner Products (<u,v>)")
    print("  →  Cosine Similarity  →  Orthogonal Projection")
    print("  →  Individual Recommendations  →  Group Blend 🎶")
    sep("═")
    print()

if __name__ == "__main__":
    main()
