"""
Semantic product search engine powered by spaCy.

This module loads a Spanish language model and a product catalog (JSON),
preprocesses titles and descriptions into embeddings and relevant lemmas,
and ranks products against a natural-language query by combining semantic
similarity, keyword overlap, and n-gram matching into a single score.
"""

import json                 # Read and write JSON files.
from pathlib import Path    # Work with portable file paths.
import spacy                # Load language models and generate NLP features.


# --------------------------------------------------------------
#  LOAD THE LANGUAGE MODEL
# --------------------------------------------------------------

try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    print("Error: Download the model with 'python -m spacy download es_core_news_md'")
    exit()


# --------------------------------------------------------------
#  SEARCH CONFIGURATION
# --------------------------------------------------------------

# Define the maximum number of n-grams to generate.
MAX_NGRAMS = 4

# Define weights for each final score component.
DEFAULT_WEIGHTS = {
    "title_similarity":     0.15,
    "description_similarity": 0.10,
    "title_match":          0.50,
    "description_match":    0.25,
    "title_ngram_match":     0.10,
    "description_ngram_match": 0.10,
}


def load_saved_weights():
    """Load the latest trained weights from the JSON history if available."""
    weights_path = Path(__file__).resolve().parent.parent / "JSON" / "best_weights.json"
    if not weights_path.exists() or weights_path.stat().st_size == 0:
        return DEFAULT_WEIGHTS

    try:
        with weights_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_WEIGHTS

    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict) and "weights" in payload:
        weights = payload["weights"]
        if isinstance(weights, dict):
            return {key: float(value) for key, value in weights.items()}
    return DEFAULT_WEIGHTS


WEIGHTS = load_saved_weights()


# --------------------------------------------------------------
#  LOAD PRODUCTS
# --------------------------------------------------------------

products_path = Path(__file__).resolve().parent.parent / "JSON" / "productos.json"
with products_path.open("r", encoding="utf-8") as file:
    products = json.load(file)


# --------------------------------------------------------------
#  HELPER FUNCTIONS
# --------------------------------------------------------------

def _generate_ngrams(text: str, max_n: int = MAX_NGRAMS):
    """Generate contiguous n-grams from a text."""
    words = text.split()
    limit = min(max_n, len(words))
    return [
        " ".join(words[start:start + n])
        for n in range(2, limit + 1)
        for start in range(len(words) - n + 1)
    ]

def _normalize_text(text: str):
    """Normalize whitespace and casing in a text."""
    return " ".join(text.lower().split())

def _calculate_ngram_match(query_ngrams, normalized_text):
    """Calculate the weighted proportion of query n-grams found in a text."""
    if not query_ngrams:
        return 0.0

    total_weight = sum(len(ngram.split()) for ngram in query_ngrams)
    found_weight = sum(
        len(ngram.split())
        for ngram in query_ngrams
        if ngram in normalized_text
    )
    return found_weight / total_weight

def _relevant_terms(doc):
    """Extract relevant non-stopword lemmas from a spaCy document."""
    return {
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"}
        and not token.is_stop
        and not token.is_punct
        and len(token.lemma_) > 2
    }


# --------------------------------------------------------------
#  PROCESS THE CATALOG
# --------------------------------------------------------------

# Preprocess titles and descriptions to obtain embeddings and relevant terms.
title_docs = list(nlp.pipe(product["titulo"].lower() for product in products))
description_docs = list(nlp.pipe(product["descripcion"].lower() for product in products))

# Extract relevant terms (lemmas) from titles and descriptions.
title_terms = [_relevant_terms(doc) for doc in title_docs]
description_terms = [_relevant_terms(doc) for doc in description_docs]

# Combine all information into tuples to simplify searching.
products_for_search = [
    (
        product["titulo"],
        product["descripcion"],
        product_title_terms,
        product_description_terms,
        title_doc,
        description_doc,
    )
    for product, title_doc, description_doc, product_title_terms, product_description_terms in zip(
        products,
        title_docs,
        description_docs,
        title_terms,
        description_terms,
    )
]


# --------------------------------------------------------------
#  MAIN SEARCH FUNCTION
# --------------------------------------------------------------

def search_products(products_details, query: str, top_k: int = 3):
    """Rank catalog products by their relevance to a natural-language query."""
    query_doc = nlp(query.lower())
    keywords = _relevant_terms(query_doc)
    if not keywords or not query_doc.vector_norm:
        return []

    normalized_query = _normalize_text(query)
    query_ngrams = _generate_ngrams(normalized_query)
    
    results = []

    for (
        title,
        description,
        title_terms,
        description_terms,
        title_doc,
        description_doc,
    ) in products_details:
        
        # Calculate semantic similarities.
        title_similarity = query_doc.similarity(title_doc)
        description_similarity = query_doc.similarity(description_doc)
        
        # Calculate keyword matches.
        title_match = len(keywords & title_terms) / len(keywords)
        description_match = len(keywords & description_terms) / len(keywords)
        
        # Calculate n-gram matches.
        title_ngram_match = _calculate_ngram_match(
            query_ngrams,
            _normalize_text(title),
        )
        description_ngram_match = _calculate_ngram_match(
            query_ngrams,
            _normalize_text(description),
        )

        final_score = (
            title_similarity * WEIGHTS["title_similarity"] +
            description_similarity * WEIGHTS["description_similarity"] +
            title_match * WEIGHTS["title_match"] +
            description_match * WEIGHTS["description_match"] +
            title_ngram_match * WEIGHTS["title_ngram_match"] +
            description_ngram_match * WEIGHTS["description_ngram_match"]
        )
        
        results.append({
            "titulo": title,
            "descripcion": description,
            "score": final_score,
            "detalles": {
                "title_similarity": round(title_similarity, 3),
                "description_similarity": round(description_similarity, 3),
                "title_match": round(title_match, 3),
                "description_match": round(description_match, 3),
                "title_ngram_match": round(title_ngram_match, 3),
                "description_ngram_match": round(description_ngram_match, 3),
            }
        })

    # Sort by descending score.
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


# --------------------------------------------------------------
#  EXECUTION EXAMPLE
# --------------------------------------------------------------

user_query = "MicroSD 512 GB"
print(f"Search: '{user_query}'\n")

best_matches = search_products(products_for_search, user_query, top_k=3)

for index, result in enumerate(best_matches, 1):
    print(f"#{index} {result['titulo']} (Score: {result['score']:.4f})")
    print(f"   Description: {result['descripcion'][:80]}...")
