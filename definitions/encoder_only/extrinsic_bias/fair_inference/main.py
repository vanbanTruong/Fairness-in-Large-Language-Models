from data import load_nli_data, mock_model_predictions
from fair_inference import evaluate_fair_inference

def main():
    data = load_nli_data()
    predictions = mock_model_predictions()
    nn, fn, t05, t07 = evaluate_fair_inference(predictions)

    print("Fair Inference Results:")
    print(f"NN: {nn:.2f}")
    print(f"FN: {fn:.2f}")
    print(f"T_0.5: {t05:.2f}")
    print(f"T_0.7: {t07:.2f}")

if __name__ == "__main__":
    main()
