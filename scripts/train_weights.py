"""Entrena los pesos del buscador semántico de productos con un AG.

El buscador conserva el preprocesamiento de ``index.py`` para que cada
evaluación reutilice los documentos y conjuntos de lemas ya calculados.
Para resultados de producción, sustituye ``queries.json`` por un golden set
etiquetado por humanos (recomendado: al menos 50-100 consultas).
"""

from __future__ import annotations          # Enables postponed evaluation of type annotations (PEP 563), allowing forward references without quotes
                                            # (blank line separating standard library imports from third-party ones)
import json                                 # Standard library for working with JSON data (encoding/decoding)
import math                                 # Standard library providing mathematical functions (e.g., sqrt, log, exp)
import random                               # Standard library for generating random numbers and random choices
import statistics                           # Standard library for statistical functions (e.g., mean, median, stdev)
import datetime                             # Standard library for working with dates and times
from pathlib import Path                    # Standard library class for object-oriented filesystem path handling
from typing import Any, Iterable, Sequence  # Type hints: Any (any type), Iterable (iterable objects), Sequence (ordered collections)
                                            # (blank line separating standard library imports from third-party ones)
import spacy                                # Third-party NLP library for natural language processing (loads language models, tokenization, etc.)


MODEL_NAME = "es_core_news_md"          # Name of the spaCy language model to load (Spanish, medium-sized, with word vectors)
MAX_NGRAMS = 4                          # Maximum size of n-grams to consider (e.g., up to 4-word sequences)
K = 3                                   # Number of items to select (e.g., top-K features/topics/documents)
SEED = 42                               # Random seed for reproducibility of results
POP_SIZE = 50                           # Size of the population in the genetic algorithm
TOURNAMENT_SIZE = 3                     # Number of individuals competing in each tournament selection
GENERATIONS = 40                        # Number of generations (iterations) to run the genetic algorithm
ELITE = 3                               # Number of best individuals preserved unchanged each generation (elitism)
MUT_PROB = 0.35                         # Probability that an individual undergoes mutation
MUTATION_SIGMA = 0.12                   # Standard deviation used for Gaussian mutation of genes
RANDOM_IMMIGRANT_RATE = 0.10            # Fraction of the population replaced by random new individuals each generation
STAGNATION_LIMIT = 8                    # Number of generations without improvement before stopping (early stopping)
IMPROVEMENT_TOLERANCE = 1e-4            # Minimum change in fitness considered an actual improvement (0.0001)

# Pesos heredados del buscador original. Su suma es 1.20; se conservan como
# baseline para no cambiar el comportamiento histórico. Todo genoma generado
# por el AG sí se normaliza obligatoriamente a suma 1.0.
WEIGHTS = {
    "title_similarity": 0.15,
    "description_similarity": 0.10,
    "title_match": 0.50,
    "description_match": 0.25,
    "title_ngram_match": 0.10,
    "description_ngram_match": 0.10,
}
WEIGHT_NAMES = tuple(WEIGHTS)

# NDCG prima el orden completo, MRR la primera posición relevante y P@K la
# densidad de resultados relevantes en el corte elegido.
FITNESS_COMPONENTS = {"ndcg": 0.5, "mrr": 0.3, "precision": 0.2}
TRAINING_QUERIES: list[dict[str, Any]] = []


def load_resources() -> tuple[Any, list[dict[str, str]]]:
    """Carga spaCy y el catálogo junto al script."""
    try:
        nlp = spacy.load(MODEL_NAME)
    except OSError as error:
        raise SystemExit(
            f"No se encontró {MODEL_NAME}. Ejecuta: python -m spacy download {MODEL_NAME}"
        ) from error
    products_path = Path(__file__).resolve().parent.parent / "JSON" / "productos.json"
    with products_path.open("r", encoding="utf-8") as file:
        products = json.load(file)
    return nlp, products


def _generate_ngrams(text: str, max_n: int = MAX_NGRAMS) -> list[str]:
    """Genera n-gramas contiguos de longitud 2 hasta ``max_n``."""
    words = text.split()
    limit = min(max_n, len(words))
    return [
        " ".join(words[start:start + n])
        for n in range(2, limit + 1)
        for start in range(len(words) - n + 1)
    ]


def _normalize_text(text: str) -> str:
    """Normaliza espacios y mayúsculas como el buscador original."""
    return " ".join(text.lower().split())


def _calculate_ngram_match(query_ngrams: Sequence[str], text: str) -> float:
    """Calcula la proporción ponderada de n-gramas presentes."""
    if not query_ngrams:
        return 0.0
    total_weight = sum(len(ngram.split()) for ngram in query_ngrams)
    found_weight = sum(
        len(ngram.split()) for ngram in query_ngrams if ngram in text
    )
    return found_weight / total_weight


def _relevant_terms(doc: Any) -> set[str]:
    """Extrae lemas relevantes usando exactamente el filtro original."""
    return {
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"}
        and not token.is_stop
        and not token.is_punct
        and len(token.lemma_) > 2
    }


def build_search_index(nlp: Any, products: list[dict[str, str]]) -> list[tuple[Any, ...]]:
    """Precomputa docs, lemas y tuplas reutilizables por cada búsqueda."""
    title_docs = list(nlp.pipe(product["titulo"].lower() for product in products))
    description_docs = list(
        nlp.pipe(product["descripcion"].lower() for product in products)
    )
    title_terms = [_relevant_terms(doc) for doc in title_docs]
    description_terms = [_relevant_terms(doc) for doc in description_docs]
    return [
        (
            product["titulo"],
            product["descripcion"],
            product_title_terms,
            product_description_terms,
            title_doc,
            description_doc,
        )
        for product, title_doc, description_doc, product_title_terms,
        product_description_terms in zip(
            products, title_docs, description_docs, title_terms, description_terms
        )
    ]

def normalize_genome(genome: Sequence[float]) -> list[float]:
    """Recorta a [0, 1] y normaliza para que la suma sea exactamente 1.0."""
    clipped = [max(0.0, min(1.0, float(value))) for value in genome]
    total = sum(clipped)
    if total <= 0.0:
        return [1.0 / len(clipped)] * len(clipped)
    normalized = [value / total for value in clipped]
    normalized[-1] = 1.0 - sum(normalized[:-1])
    return normalized

def make_initial_population(rng: random.Random) -> list[list[float]]:
    """Crea una población totalmente aleatoria y normalizada."""
    population = [
        normalize_genome([rng.random() for _ in WEIGHT_NAMES])
        for _ in range(POP_SIZE)
    ]
    return population

def crossover(
    parent_a: Sequence[float], parent_b: Sequence[float], rng: random.Random
) -> list[float]:
    """Cruce aritmético: mezcla cada gen con un alpha aleatorio."""
    child = [
        alpha * value_a + (1.0 - alpha) * value_b
        for value_a, value_b, alpha in zip(
            parent_a, parent_b, (rng.random() for _ in parent_a)
        )
    ]
    return normalize_genome(child)

def mutate(
    genome: Sequence[float], rng: random.Random, probability: float = MUT_PROB,
    sigma: float = MUTATION_SIGMA,
) -> list[float]:
    """Muta varios genes, garantizando al menos una mutación por hijo."""
    mutation_indexes = [
        index for index in range(len(genome)) if rng.random() < probability
    ]
    if not mutation_indexes:
        mutation_indexes = [rng.randrange(len(genome))]
    mutated = list(genome)
    for index in mutation_indexes:
        mutated[index] += rng.gauss(0.0, sigma)
    return normalize_genome(mutated)

def search_products(
    products_details: Sequence[tuple[Any, ...]], query: str, top_k: int = 3,
    weights: dict[str, float] = WEIGHTS,
) -> list[dict[str, Any]]:
    """Rankea productos con seis componentes y los pesos recibidos."""
    query_doc = NLP(query.lower())
    keywords = _relevant_terms(query_doc)
    if not keywords or not query_doc.vector_norm:
        return []
    normalized_query = _normalize_text(query)
    query_ngrams = _generate_ngrams(normalized_query)
    results = []
    for title, description, title_terms, description_terms, title_doc, description_doc in products_details:
        title_similarity = query_doc.similarity(title_doc)
        description_similarity = query_doc.similarity(description_doc)
        title_match = len(keywords & title_terms) / len(keywords)
        description_match = len(keywords & description_terms) / len(keywords)
        title_ngram_match = _calculate_ngram_match(query_ngrams, _normalize_text(title))
        description_ngram_match = _calculate_ngram_match(
            query_ngrams, _normalize_text(description)
        )
        components = {
            "title_similarity": title_similarity,
            "description_similarity": description_similarity,
            "title_match": title_match,
            "description_match": description_match,
            "title_ngram_match": title_ngram_match,
            "description_ngram_match": description_ngram_match,
        }
        final_score = sum(components[name] * weights[name] for name in WEIGHT_NAMES)
        results.append({
            "titulo": title,
            "descripcion": description,
            "score": final_score,
            "detalles": {name: round(value, 3) for name, value in components.items()},
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]

def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int = K) -> float:
    """Calcula Precision@K sobre títulos."""
    selected = list(retrieved[:k])
    return sum(title in relevant for title in selected) / k if k else 0.0

def mrr_at_k(retrieved: Sequence[str], relevant: set[str], k: int = K) -> float:
    """Calcula Reciprocal Rank truncado en K."""
    for rank, title in enumerate(retrieved[:k], start=1):
        if title in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int = K) -> float:
    """Calcula NDCG@K con relevancia binaria y varios ground truths."""
    gains = [1.0 if title in relevant else 0.0 for title in retrieved[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0

def evaluate_query(
    products_details: Sequence[tuple[Any, ...]], query_data: dict[str, Any],
    weights: dict[str, float], k: int = K,
) -> dict[str, float]:
    """Busca una consulta y devuelve sus métricas y fitness combinado."""
    results = search_products(products_details, query_data["query"], k, weights)
    retrieved = [result["titulo"] for result in results]
    relevant = set(query_data["relevant_titles"])
    metrics = {
        "precision": precision_at_k(retrieved, relevant, k),
        "mrr": mrr_at_k(retrieved, relevant, k),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }
    metrics["fitness"] = sum(
        FITNESS_COMPONENTS[name] * metrics[name] for name in FITNESS_COMPONENTS
    )
    return metrics

def _load_queries() -> list[dict[str, Any]]:
    """Lee etiquetas y crea 12 formulaciones reproducibles por consulta.

    Las variantes aumentan cobertura lingüística del ejemplo. En producción,
    reemplaza el archivo por consultas distintas etiquetadas por humanos.
    """
    path = Path(__file__).resolve().parent.parent / "JSON" / "queries.json"
    with path.open("r", encoding="utf-8") as file:
        queries = json.load(file)
    if len(queries) < 20:
        raise ValueError("queries.json debe contener al menos 20 consultas etiquetadas")
    prefixes = (
        "",
        "necesito",
        "busco",
        "quiero comprar",
        "recomiéndame",
        "me interesa",
        "una opción de",
        "mejor opción para",
        "producto para",
        "estoy buscando",
        "necesito encontrar",
        "qué producto sirve para",
    )
    expanded = []
    for item in queries:
        for prefix in prefixes:
            query = item["query"] if not prefix else f"{prefix} {item['query']}"
            expanded.append({
                "query": query,
                "relevant_titles": list(item["relevant_titles"]),
            })
    return expanded

def _metrics_for_queries(
    products_details: Sequence[tuple[Any, ...]], queries: Iterable[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, float]:
    """Promedia métricas de un conjunto de consultas."""
    values = [evaluate_query(products_details, query, weights) for query in queries]
    return {
        name: statistics.fmean(item[name] for item in values)
        for name in ("ndcg", "mrr", "precision", "fitness")
    }

def evaluate_individual(
    genome: Sequence[float], products_details: Sequence[tuple[Any, ...]],
    queries: Sequence[dict[str, Any]],
) -> float:
    """Calcula fitness medio y penaliza una distribución extremadamente degenerada."""
    weights = dict(zip(WEIGHT_NAMES, normalize_genome(genome)))
    fitness = _metrics_for_queries(products_details, queries, weights)["fitness"]
    entropy = -sum(weight * math.log(weight) for weight in weights.values() if weight > 0)
    maximum_entropy = math.log(len(weights))
    if entropy < maximum_entropy * 0.10:
        fitness -= 0.01 * (1.0 - entropy / maximum_entropy)
    return fitness

def _tournament_selection(
    population: Sequence[list[float]], scores: dict[int, float], rng: random.Random
) -> list[float]:
    """Selecciona un individuo mediante torneo configurable de tamaño 3."""
    candidates = rng.sample(list(population), TOURNAMENT_SIZE)
    return max(candidates, key=lambda genome: scores[id(genome)])

def genetic_algorithm(
    products_details: Sequence[tuple[Any, ...]],
    train_queries: Sequence[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[float], float, float, int]:
    """Optimiza los seis pesos con torneo, cruce, mutación y elitismo."""
    population = make_initial_population(rng)
    best_genome = population[-1]
    best_fitness = float("-inf")
    mean_fitness = float("-inf")
    stagnant_generations = 0
    completed_generations = 0
    for generation in range(1, GENERATIONS + 1):
        scored = [
            (evaluate_individual(genome, products_details, train_queries), genome)
            for genome in population
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        fitnesses = [item[0] for item in scored]
        current_fitness, current_genome = scored[0]
        mean_fitness = statistics.fmean(fitnesses)
        best_weights = dict(zip(WEIGHT_NAMES, current_genome))
        print(
            f"gen={generation:02d}: best={current_fitness:.4f} "
            f"; mean={statistics.fmean(fitnesses):.4f} "
            f"; std={statistics.pstdev(fitnesses):.4f} "
            f"; {{ ts={best_weights['title_similarity']:.4f} , "
            f"ds={best_weights['description_similarity']:.4f} , "
            f"tm={best_weights['title_match']:.4f} , "
            f"dm={best_weights['description_match']:.4f} , "
            f"tn={best_weights['title_ngram_match']:.4f} , "
            f"dn={best_weights['description_ngram_match']:.4f} }}"
        )
        completed_generations = generation
        if current_fitness > best_fitness + IMPROVEMENT_TOLERANCE:
            best_fitness = current_fitness
            best_genome = list(current_genome)
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        if stagnant_generations >= STAGNATION_LIMIT:
            break
        population_scores = {
            id(genome): score for score, genome in scored
        }
        next_population = [list(genome) for _, genome in scored[:ELITE]]
        while len(next_population) < POP_SIZE:
            if rng.random() < RANDOM_IMMIGRANT_RATE:
                next_population.append(
                    normalize_genome([rng.random() for _ in WEIGHT_NAMES])
                )
                continue
            parent_a = _tournament_selection(population, population_scores, rng)
            parent_b = _tournament_selection(population, population_scores, rng)
            next_population.append(mutate(crossover(parent_a, parent_b, rng), rng))
        population = next_population
    return normalize_genome(best_genome), best_fitness, mean_fitness, completed_generations

def save_best_weights(
    weights: dict[str, float], best: float, mean: float,
    generations: int,
) -> None:
    """Guarda el resultado del entrenamiento al principio del historial."""
    output_path = Path(__file__).resolve().parent.parent / "JSON" / "best_weights.json"
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training = {
        "weights": weights,
        "best": best,
        "mean": mean,
        "generations": generations,
        "date": date,
    }
    history = []
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
        if isinstance(history, dict):
            history = [history]
        for previous_training in history:
            if "best" not in previous_training and "fitness_train" in previous_training:
                previous_training["best"] = previous_training.pop("fitness_train")
            else:
                previous_training.pop("fitness_train", None)
            if "mean" not in previous_training and "fitness_test" in previous_training:
                previous_training["mean"] = previous_training.pop("fitness_test")
            else:
                previous_training.pop("fitness_test", None)
            previous_training.pop("seed", None)
            if "date" not in previous_training:
                previous_training["date"] = datetime.datetime.fromtimestamp(
                    output_path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, training)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    print(
        f"Entrenamiento guardado con fecha {date}. "
        f"Historial total: {len(history)} entrenamientos."
    )

def _metrics_for_queries_and_print(
    products_details: Sequence[tuple[Any, ...]], train_queries: Sequence[dict[str, Any]],
    test_queries: Sequence[dict[str, Any]], trained_weights: dict[str, float],
) -> None:
    """Imprime pesos, métricas y tres búsquedas comparativas."""
    original_train = _metrics_for_queries(products_details, train_queries, WEIGHTS)
    original_test = _metrics_for_queries(products_details, test_queries, WEIGHTS)
    trained_train = _metrics_for_queries(products_details, train_queries, trained_weights)
    trained_test = _metrics_for_queries(products_details, test_queries, trained_weights)
    print("\nComparación de pesos y métricas")
    print(f"{'componente':<25} {'original':>12} {'entrenado':>12}")
    for name in WEIGHT_NAMES:
        print(f"{name:<25} {WEIGHTS[name]:>12.6f} {trained_weights[name]:>12.6f}")
    print("\n                 NDCG       MRR       P@K")
    for label, metrics in (
        ("Original train", original_train), ("Entrenado train", trained_train),
        ("Original test", original_test), ("Entrenado test", trained_test),
    ):
        print(f"{label:<16} {metrics['ndcg']:.4f}    {metrics['mrr']:.4f}    {metrics['precision']:.4f}")
    print("\nEjemplos cualitativos")
    for query_data in test_queries[:3]:
        query = query_data["query"]
        original = search_products(products_details, query, K, WEIGHTS)
        trained = search_products(products_details, query, K, trained_weights)
        print(f"\nConsulta: {query}")
        print("  Original : " + ", ".join(item["titulo"] for item in original))
        print("  Entrenado: " + ", ".join(item["titulo"] for item in trained))

def main() -> None:
    """Carga recursos, entrena, evalúa y guarda los pesos."""
    global NLP, TRAINING_QUERIES
    print("Cargando modelo spaCy...")
    NLP, products = load_resources()
    print("Cargando catálogo de productos...")
    products_for_search = build_search_index(NLP, products)
    print("Cargando y expandiendo consultas...")
    TRAINING_QUERIES = _load_queries()
    rng = random.Random(SEED)
    rng.shuffle(TRAINING_QUERIES)
    print("Dividiendo train/test...")
    split = max(1, int(len(TRAINING_QUERIES) * 0.7))
    train_queries, test_queries = TRAINING_QUERIES[:split], TRAINING_QUERIES[split:]
    print("Generando población inicial...")
    print("Iniciando evolución...")
    print(
        "Leyenda: ts=title_similarity, ds=description_similarity, "
        "tm=title_match, dm=description_match, tn=title_ngram_match, "
        "dn=description_ngram_match\n"
    )
    trained_genome, best, mean, generations = genetic_algorithm(
        products_for_search, train_queries, rng
    )
    trained_weights = dict(zip(WEIGHT_NAMES, trained_genome))
    print("Evaluación final...")
    print("Guardando pesos en best_weights.json...")
    save_best_weights(
        trained_weights, best, mean, generations,
    )
    _metrics_for_queries_and_print(
        products_for_search, train_queries, test_queries, trained_weights
    )
    print("\nPesos guardados en best_weights.json")

NLP: Any = None

if __name__ == "__main__":
    main()