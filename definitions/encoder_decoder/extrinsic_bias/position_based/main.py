from data import load_articles_and_summaries
from position_disparity import compute_npd

def main():
    dataset = load_articles_and_summaries()
    npd_score = compute_npd(dataset, segments=3)

    print("Position-based Disparity Evaluation (Encoder-Decoder):")
    print(f"Normalized Position Disparity (NPD): {npd_score}")

if __name__ == "__main__":
    main()
