from data import load_counterfactual_pairs, get_predictions, get_token_outputs
from counterfactual_fairness import compute_cr, compute_ctf

def main():
    pairs = load_counterfactual_pairs()
    preds = get_predictions()
    token_scores = get_token_outputs()

    cr_score = compute_cr(pairs, preds)
    ctf_score = compute_ctf(token_scores)

    print("Counterfactual Fairness Results (Decoder-only):")
    print(f"CR Score: {cr_score}")
    print(f"CTF Score: {ctf_score}")

if __name__ == "__main__":
    main()
