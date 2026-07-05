# Not_JUST_ANOTHER_movie_recommendation

A movie recommendation app using **Singular Value Decomposition (SVD)**.

---

## 📖 Reference Article

You can check out the idea from this article:

🔗 https://machinelearningmastery.com/using-singular-value-decomposition-to-build-a-recommender-system/

---

## 🌐 Live Demo

Website:

🔗 https://notjustanothermovierecommendation-euehfg38scs2yyrevu9dmw.streamlit.app/

You can try it! The dataset mostly contains some older movies, which can be fun to watch!!

---

# 📊 About the Dataset

## Ratings File Description

All ratings are contained in the file **`ratings.dat`** and are in the following format:

```text
UserID::MovieID::Rating::Timestamp
```

- UserIDs range between **1 and 6040**
- MovieIDs range between **1 and 3952**
- Ratings are made on a **5-star scale** (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by `time(2)`
- Each user has at least **20 ratings**

---

# ⚙️ Training Process

It's simple:

1. Convert the ratings DataFrame into a matrix.
2. Perform **Compact SVD**.

The matrix decomposition is:

```text
M = U · Σ · Vᵀ
```

## What does SVD mean for the ratings matrix?

Imagine we collected some book reviews such that:

- **Rows** represent people.
- **Columns** represent books.
- **Entries** represent the ratings a person gave to a book.

In that case:

- **M · Mᵀ** would produce a **person-to-person similarity matrix**, where the entries represent how similar two users are based on their ratings.
- **Mᵀ · M** would produce a **book-to-book similarity matrix**, where the entries represent how similar two books are based on the ratings they received.

What could be the hidden connection between people and books?

It could be:

- Genre
- Author
- Or something of a similar nature

It kind of works like a neural network, but for less complex data.

So,

```text
Σ · Vᵀ
```

gives us our model for **book similarities**.

You can read the full explanation here:

🔗 https://machinelearningmastery.com/using-singular-value-decomposition-to-build-a-recommender-system/

---

# 🔍 One More Observation

I have used only the **first 100 dominant singular values**.

So, Compact SVD removes noise from the dataset.

For example, a user may like every movie they watch—not because of a meaningful preference, but simply due to their personal rating behavior. If other users don't share that pattern, we effectively ignore (i.e., assign less weight to) that single user's preference.

---

# 📓 Training Notebook

Here is the Kaggle notebook where I trained and preprocessed the DataFrames according to my needs.

```
https://www.kaggle.com/code/bhavyasharma2005/not-just-another-recommendation
```


## 🤝 Contributing / Suggestions

I would really appreciate suggestions and feedback on this project.

Some areas where I'd especially love input:

- The dataset contains additional user information that I haven't used yet. If you have ideas on how it could improve the recommendation system, I'd love to hear them.
- I currently use the **first 100 dominant singular values**. I'm not sure if this is the optimal choice, so suggestions on selecting the appropriate number of singular values are welcome.
- I'm still learning data analysis and recommendation algorithms, so if you notice a better approach, optimization, or evaluation method, please let me know.

You can reach me at:

📧 **sharmabhavya978@gmail.com**
