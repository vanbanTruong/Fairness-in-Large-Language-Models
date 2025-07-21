from data import load_translation_outputs, get_token_frequencies, get_morphological_variants
from algorithmic_disparity import compute_lfp, compute_entropy, compute_simpson

def main():
    outputs = load_translation_outputs()
    freqs = get_token_frequencies()
    variants = get_morphological_variants()

    p_b1, p_b2, p_b3 = compute_lfp(outputs, freqs)
    entropy_score = compute_entropy(variants)
    simpson_score = compute_simpson(variants)

    print("Algorithmic Disparity Results (Encoder-Decoder):")
    print(f"LFP - B1: {p_b1}, B2: {p_b2}, B3: {p_b3}")
    print(f"MCD - Entropy (H): {entropy_score}")
    print(f"MCD - Simpson Diversity (D): {simpson_score}")

if __name__ == "__main__":
    main()
