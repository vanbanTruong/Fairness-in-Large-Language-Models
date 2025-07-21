from data import load_hypothesis_pairs
from fair_inference import compute_ibs_score

def main():
    data = load_hypothesis_pairs()
    score = compute_ibs_score(data)

    print("Fair Inference Bias Evaluation (Encoder–Decoder LMs)")
    print(f"Inference Bias Score (IBS): {score}")
    print("Model: mBART | Datasets: XNLI, XSum, WinoMT")

if __name__ == "__main__":
    main()
