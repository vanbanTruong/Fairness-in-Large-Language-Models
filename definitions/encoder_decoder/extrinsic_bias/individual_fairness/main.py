from data import load_translation_pairs
from individual_fairness import ss_score

def main():
    translation_pairs = load_translation_pairs()
    score = ss_score(translation_pairs)

    print("Individual Fairness Evaluation (Encoder–Decoder):")
    print("Metric: Semantic Similarity (SS)")
    print(f"SS Score: {score}")

if __name__ == "__main__":
    main()
