import glob
import pandas as pd
from encoder_only.extrinsic.equal_opportunity.data import load_classification_data, extract_instances_by_group
from encoder_only.extrinsic.equal_opportunity.gap import compute_tpr, compute_gap

def run_experiment():
    print("---- Encoder-only LMs: Extrinsic Bias – Equal Opportunity (GAP) ----")
    result_list = []
    dataset_paths = glob.glob("data/classification/*.json")

    for path in dataset_paths:
        data = load_classification_data(path)
        grouped_counts = extract_instances_by_group(data)
        tprs = compute_tpr(grouped_counts)
        gaps = compute_gap(tprs)

        for entry in gaps:
            entry["Dataset"] = path.split("/")[-1].replace(".json", "")
            result_list.append(entry)

    df = pd.DataFrame(result_list)
    df.to_csv("encoder_only/extrinsic/equal_opportunity/result.csv", index=False)

if __name__ == "__main__":
    run_experiment()
