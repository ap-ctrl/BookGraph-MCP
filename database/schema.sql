-- Romance Reader Intelligence SQLite Schema
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS books (
    book_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    title_without_series TEXT NOT NULL,
    author_name TEXT NOT NULL,
    author_id TEXT,
    average_rating REAL NOT NULL DEFAULT 0.0,
    ratings_count INTEGER NOT NULL DEFAULT 0,
    text_reviews_count INTEGER NOT NULL DEFAULT 0,
    publication_year INTEGER,
    publisher TEXT,
    num_pages INTEGER,
    description TEXT,
    genres TEXT NOT NULL DEFAULT '["Romance"]',
    popular_shelves TEXT,
    similar_books TEXT,
    url TEXT,
    image_url TEXT,
    bayesian_rating REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_books_title ON books(title COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_books_author ON books(author_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_books_rating ON books(average_rating DESC);
CREATE INDEX IF NOT EXISTS idx_books_bayesian ON books(bayesian_rating DESC);
CREATE INDEX IF NOT EXISTS idx_books_year ON books(publication_year);

CREATE TABLE IF NOT EXISTS interactions (
    user_id TEXT NOT NULL,
    book_id TEXT NOT NULL,
    rating INTEGER NOT NULL DEFAULT 0,
    is_read INTEGER NOT NULL DEFAULT 1,
    date_added TEXT,
    read_at TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, book_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_book ON interactions(book_id);
CREATE INDEX IF NOT EXISTS idx_interactions_rating ON interactions(rating);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    book_id TEXT NOT NULL,
    rating INTEGER NOT NULL DEFAULT 0,
    review_text TEXT NOT NULL,
    sentiment_score REAL,
    sentiment_label TEXT,
    date_added TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reviews_book ON reviews(book_id);
CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label);

CREATE TABLE IF NOT EXISTS tropes (
    book_id TEXT NOT NULL,
    trope_name TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    method TEXT NOT NULL,
    evidence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (book_id, trope_name),
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tropes_name ON tropes(trope_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_tropes_book ON tropes(book_id);

CREATE TABLE IF NOT EXISTS review_themes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    theme_type TEXT NOT NULL, -- 'character', 'pacing', 'plot'
    theme_key TEXT NOT NULL,
    score REAL NOT NULL,
    evidence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_themes_book ON review_themes(book_id);
CREATE INDEX IF NOT EXISTS idx_themes_type ON review_themes(theme_type);

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    k INTEGER NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    ndcg REAL NOT NULL,
    coverage REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
