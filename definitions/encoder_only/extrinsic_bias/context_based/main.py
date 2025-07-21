from data import load_qa_contexts, mock_model_outputs
from context_disparity import compute_s_dis, compute_s_amb

def main():
    contexts = load_qa_contexts()
    outputs = mock_model_outputs()

    s_dis_score = compute_s_dis(outputs)
    s_amb_score = compute_s_amb(outputs, s_dis_score)

    print("Context-based Disparity Results:")
    print(f"S_DIS: {s_dis_score:.2f}")
    print(f"S_AMB: {s_amb_score:.2f}")

if __name__ == "__main__":
    main()
