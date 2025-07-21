from data import load_sentences, get_attention_weights, get_seat_embedding_scores
from attention_disparity import compute_nie_bias, compute_gbe_gradient

def main():
    sentences = load_sentences()
    weights = get_attention_weights()
    seat_data = get_seat_embedding_scores()

    nie_score = compute_nie_bias(sentences, weights)
    gbe_score = compute_gbe_gradient(seat_data)

    print("Attention Head-based Disparity Results (Decoder-only):")
    print(f"NIE Score: {nie_score}")
    print(f"GBE Score: {gbe_score}")

if __name__ == "__main__":
    main()
