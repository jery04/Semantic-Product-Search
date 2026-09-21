# 🛒 Semantic Product Search

Python project that uses spaCy and the `es_core_news_md` model to search products by semantic similarity, term matching, and n-gram matching. It also includes a script to train the search weights using a genetic algorithm.

## 📁 Project structure

```text
.
├── JSON/
│   ├── best_weights.json   # History of trained weights
│   ├── productos.json       # Product catalog
│   └── queries.json         # Labeled queries for training
├── scripts/
│   ├── index.py             # Semantic product search
│   └── train_weights.py     # Weight training and evaluation
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── Taskfile.yml             # Automated tasks
├── .gitignore               # Environment and temporary file exclusions
└── .venv/                   # Local virtual environment, ignored by Git
```

## ⚙️ What each script does

### 🔎 `scripts/index.py`

Loads the catalog from `JSON/productos.json`, processes the text with spaCy, and runs a sample query (`MicroSD 512 GB`). It shows the top three ranked products along with their score and a breakdown of the score components.

### 🧬 `scripts/train_weights.py`

Loads the catalog and labeled queries, splits the queries into training and test sets, and optimizes the six search weights using a genetic algorithm. It saves each result in `JSON/best_weights.json` and shows a comparison between the original and trained weights.

## 📦 Installing dependencies (with Taskfile)

From the project root, first create the environment and install everything with:

```bash
task install
```

This task creates `.venv`, installs the dependencies from `requirements.txt` in that environment, and downloads the Spanish spaCy model inside the same environment.

## ▶️ Running the project

Run the commands from the project root after installing the dependencies.

### 🔍 Search

```bash
task search
```

### 🧠 Weight training

```bash
task train
```

The training step updates `JSON/best_weights.json` with a new result entry.
