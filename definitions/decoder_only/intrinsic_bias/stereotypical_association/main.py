from data import load_prompts, get_concept_outputs, get_reference_distribution
from stereotypical_association import compute_sll, compute_ca

def main():
    prompts = load_prompts()
    concepts = get_concept_outputs()
    ref = get_reference_distribution()

    sll_score = compute_sll(prompts)
    ca_score = compute_ca(concepts, ref)

    print("Stereotypical Association Results (Decoder-only):")
    print(f"SLL Score: {sll_score}")
    print(f"CA Score: {ca_score}")

if __name__ == "__main__":
    main()
