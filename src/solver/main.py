# src/main.py

# src/solver/main.py
import argparse
from src.utils.read_instance    import read_instance_auto
from src.utils.instance         import Instance
from src.algorithms.localsearch import LocalSearchPCenter


def main():
    parser = argparse.ArgumentParser(
        description="Résolution du problème P-Center par recherche locale"
    )
    parser.add_argument(
        "instance_file",
        help="Chemin vers le fichier d’instance (format matrice ou graphe)"
    )
    args = parser.parse_args()

    # Lecture automatique de l'instance
    n, p, dist = read_instance_auto(args.instance_file)
    print(f"[INFO] Instance chargée : n={n}, p={p}")

    # Création de l’objet Instance
    instance = Instance(n, p, dist)

    # Lancement de la recherche locale
    ls = LocalSearchPCenter(instance)
    solution, cost = ls.search(verbose=True)

    # Affichage du résultat
    print("\n=== Résultat ===")
    print(f"Centres choisis : {sorted(solution)}")
    print(f"Coût maximum    : {cost:.2f}")

if __name__ == "__main__":
    main()
