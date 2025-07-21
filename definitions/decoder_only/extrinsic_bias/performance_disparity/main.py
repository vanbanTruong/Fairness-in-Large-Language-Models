from data import load_performance_data, load_biasasker_data, load_sns_data
from performance_disparity import compute_accuracy_disparity, compute_biasasker_score, compute_sns

def main():
    performance_data = load_performance_data()
    biasasker_data = load_biasasker_data()
    sns_data = load_sns_data()

    ad_score = compute_accuracy_disparity(performance_data)
    ab_score, rb_score = compute_biasasker_score(biasasker_data)
    snsr_score, snsv_score = compute_sns(sns_data)

    print("Performance Disparity Metrics Evaluation:")
    print(f"Accuracy Disparity (AD): {ad_score}")
    print(f"BiasAsker Absolute Bias (AB): {ab_score}")
    print(f"BiasAsker Relative Bias (RB): {rb_score}")
    print(f"Sensitive-to-Neutral Similarity Range (SNSR): {snsr_score}")
    print(f"Sensitive-to-Neutral Similarity Variance (SNSV): {snsv_score}")

if __name__ == "__main__":
    main()
