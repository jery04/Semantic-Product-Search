#!/usr/bin/env python3
"""Muestra el estado del checkpoint de entrenamiento actual."""

import json
from pathlib import Path

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "JSON" / "best_weights_checkpoint.json"
WEIGHT_NAMES = ("title_similarity", "description_similarity", "title_match", 
                "description_match", "title_ngram_match", "description_ngram_match")

def main():
    if not CHECKPOINT_PATH.exists():
        print("❌ No hay checkpoint guardado aún.")
        print("   Inicia el entrenamiento con: python scripts/train_weights.py")
        return
    
    try:
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
            checkpoint = json.load(file)
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error al leer checkpoint: {e}")
        return
    
    print("\n" + "="*60)
    print("📊 ESTADO DEL CHECKPOINT DE ENTRENAMIENTO")
    print("="*60)
    
    generations = checkpoint.get("generations", 0)
    best_fitness = checkpoint.get("best", 0)
    mean_fitness = checkpoint.get("mean", 0)
    date = checkpoint.get("date", "desconocida")
    weights = checkpoint.get("weights", {})
    
    print(f"\n✓ Generación alcanzada: {generations}")
    print(f"✓ Mejor fitness: {best_fitness:.4f}")
    print(f"✓ Fitness promedio: {mean_fitness:.4f}")
    print(f"✓ Fecha del checkpoint: {date}")
    
    if weights:
        print(f"\n📈 Pesos optimizados:")
        print(f"{'Componente':<30} {'Peso':>10}")
        print("-" * 42)
        total = 0
        for name in WEIGHT_NAMES:
            value = weights.get(name, 0)
            print(f"  {name:<28} {value:>10.6f}")
            total += value
        print("-" * 42)
        print(f"  {'Total':<28} {total:>10.6f}")
    
    remaining_gens = 40 - generations
    print(f"\n⏳ Generaciones restantes para completar 40: {remaining_gens}")
    
    print(f"\n💡 Comandos útiles:")
    print(f"   • Reanudar entrenamiento: python scripts/train_weights.py --resume")
    print(f"   • Ver estado nuevamente:  python scripts/checkpoint_status.py")
    print(f"   • Entrenar sin reanudar:  python scripts/train_weights.py")
    print()

if __name__ == "__main__":
    main()
