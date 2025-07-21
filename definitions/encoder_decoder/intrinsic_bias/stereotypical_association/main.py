from data import get_stereotypical_examples, get_antistereotypical_examples, get_attention_heads
from stereotypical_association import compute_sd_metric, compute_sva_shapley

def main():
    stereo = get_stereotypical_examples()
    anti = get_antistereotypical_examples()
    heads = get_attention_heads()

    sd_score = compute_sd_metric(stereo, anti)
    sva_score = compute_sva_shapley(heads)

    print("Stereotypical Association Results (Encoder-Decoder):")
    print(f"SD (ΔS): {sd_score}")
    print(f"SVA (ϕ): {sva_score}")

if __name__ == "__main__":
    main()
