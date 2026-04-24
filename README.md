# 🎵 Song Recommendation System — Linear Algebra Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy)
![Linear Algebra](https://img.shields.io/badge/Linear_Algebra-Norms_%26_Inner_Products-2E75B6?style=for-the-badge)
![PES University](https://img.shields.io/badge/PES_University-Mini_Project-orange?style=for-the-badge)

**UE24MA241B – Linear Algebra and Its Applications**
**Department of Computer Science and Engineering | PES University | 2024–25**

</div>

---

## 📌 About the Project

Ever wondered how **Spotify** knows exactly what song to play next?

This project builds a fully functional **Song Recommendation System** from scratch using **pure Linear Algebra** — no machine learning libraries, no black boxes. Just math.

Users rate 6 songs across genres. The system then uses **vectors, norms, inner products, cosine similarity, and orthogonal projection** to measure taste similarity between users and recommend songs they have never heard before.

Two types of recommendations are produced:
- 🎧 **Individual** — the best song for each person based on their favourite genre
- 🍿 **Group Blend** — one song that all 4 friends would enjoy together

---

## 👥 Team

| Name | Role |
|------|------|
| Affan | Developer |
| Atharv | Developer |
| Rizwan | Developer |
| Mohit | Developer |

---

## 🧮 Linear Algebra Pipeline

```
Real-World Data
      ↓
Matrix A (4×6)        →   rows = users, columns = genres
      ↓
User Vectors in R⁶    →   each row is a 6D taste vector
      ↓
Norms                 →   ||v|| = √(Σvᵢ²)  measures rating intensity
      ↓
Inner Products        →   ⟨u,v⟩ = Σuᵢvᵢ   measures raw similarity
      ↓
Cosine Similarity     →   cos(θ) = ⟨u,v⟩ / (||u|| × ||v||)
      ↓
Compatibility Scores  →   all 6 friend pairs ranked
      ↓
Orthogonal Projection →   predict ratings for 18 unseen songs
      ↓
Individual Recs  +  Group Blend 🎶
```

---

## 🎵 Songs in the System

### Songs Users Rate (1 per genre)

| Genre | Song |
|-------|------|
| Pop | Blinding Lights – The Weeknd |
| Hip-Hop | HUMBLE. – Kendrick Lamar |
| Rock | Bohemian Rhapsody – Queen |
| R&B | No One – Alicia Keys |
| EDM | Levels – Avicii |
| Indie | Do I Wanna Know? – Arctic Monkeys |

### Catalogue Songs (System Recommends From These)

| Genre | Song 1 | Song 2 | Song 3 |
|-------|--------|--------|--------|
| Pop | As It Was – Harry Styles | Anti-Hero – Taylor Swift | Stay – The Kid LAROI |
| Hip-Hop | God's Plan – Drake | Sicko Mode – Travis Scott | Rockstar – Post Malone |
| Rock | Mr. Brightside – The Killers | Smells Like Teen Spirit – Nirvana | Thunder – Imagine Dragons |
| R&B | Golden – Joji | Redbone – Childish Gambino | Save Your Tears – The Weeknd |
| EDM | Titanium – David Guetta | Lean On – Major Lazer | Animals – Martin Garrix |
| Indie | Fluorescent Adolescent – Arctic Monkeys | 505 – Arctic Monkeys | Sweater Weather – The Neighbourhood |

---

## 🚀 How to Run

**1. Make sure Python and NumPy are installed**
```bash
pip install numpy
```

**2. Run the program**
```bash
python song_recommender.py
```

**3. Each of the 4 users rates 6 songs (scale 1–5)**

The system then outputs:
- ✅ User–Genre Rating Matrix
- ✅ Norms for each user
- ✅ Inner Product Matrix
- ✅ Cosine Similarity Matrix
- ✅ Friend Compatibility Scores (all 6 pairs)
- ✅ Individual song recommendation per user
- ✅ Group Blend recommendation for all 4 friends

---

## 📊 Sample Output

```
══════════════════════════════════════════════════════════════
  🎵  SONG RECOMMENDATION SYSTEM
  📐  Powered by Linear Algebra (Norms & Inner Products)
  👥  Users: Affan | Atharv | Rizwan | Mohit
══════════════════════════════════════════════════════════════

  🤝  FRIEND COMPATIBILITY SCORES
  ────────────────────────────────────────────────────────
  Affan    ↔ Mohit     [███████████████████░]   97.6%  🔥 Best match!
  Atharv   ↔ Mohit     [█████████████████░░░]   87.2%  🔥 Best match!
  Affan    ↔ Atharv    [█████████████████░░░]   87.0%  🔥 Best match!

  👤  Affan
      Loved genre    : Pop
      Recommended    : 🎵 As It Was – Harry Styles
      Predicted score: 4.20 / 5.00

  🎵  Best song for the group: "Golden – Joji"
      Average predicted rating : 4.15 / 5.00
```

---

## 📐 Math Behind It

### Norm
Measures the magnitude of a user's rating vector:
```
||v|| = √(v₁² + v₂² + v₃² + v₄² + v₅² + v₆²)
```

### Inner Product
Measures raw similarity between two users:
```
⟨u, v⟩ = u₁v₁ + u₂v₂ + u₃v₃ + u₄v₄ + u₅v₅ + u₆v₆
```

### Cosine Similarity
Normalised similarity — removes rating scale bias:
```
cos(θ) = ⟨u, v⟩ / (||u|| × ||v||)
```

### Orthogonal Projection
Predicts unseen ratings using nearest neighbours:
```
predicted = Σ (similarity × neighbour_genre_rating)
            ─────────────────────────────────────────
                       Σ (similarity)
```

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Library:** NumPy (matrix operations, norms, dot products)
- **Concepts:** Linear Algebra — Matrices, Vectors, Norms, Inner Products, Cosine Similarity, Orthogonal Projection

---

## 📁 Project Structure

```
song-recommender-linear-algebra/
│
├── song_recommender.py     ← Main program (run this)
└── README.md               ← Project documentation
```

---

<div align="center">
Made with 📐 Linear Algebra | PES University 2024–25
</div>
