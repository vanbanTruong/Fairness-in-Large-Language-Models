from data import load_generated_outputs, load_probabilities, get_label_sets
from demographic_representation import compute_drd, compute_dnp

def main():
    outputs = load_generated_outputs()
    labels = get_label_sets()
    probs = load_probabilities()

    drd_score = compute_drd(outputs, labels)
    print("Demographic Representation Disparity (DRD) Score:")
    print(f"→ DRD: {drd_score}")

    print("\nDemographic Normalized Probability (DNP) Scores:")
    for dataset, p in probs.items():
        s, s_dash, d = compute_dnp(p)
        print(f"{dataset}:")
        print(f"  - P_s: {s}")
        print(f"  - P_s': {s_dash}")
        print(f"  - P_d: {d}")

if __name__ == "__main__":
    main()
