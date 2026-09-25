# The taxonomy

`fairLMs/definitions/` is both the public surface and the single implementation
tree. Its root exports 33 metric classes grouped by family, each exposing
`compute(model, data) -> MetricResult`. The architecture directories below it
organise the low-level calculations along the two axes used in the literature:
model architecture and where the bias is measured.

```text
fairLMs/definitions/
├── encoder_only/
│   ├── intrinsic_bias/
│   │   ├── similarity_based/            weat, seat, ceat
│   │   └── probability_based/
│   │       ├── pseudo_log_likelihood_metrics/   cps, pll, aul, aula, cat
│   │       └── masked_token_metrics/            lpbs, disco, cbs
│   └── extrinsic_bias/                  fair_inference, equal_opportunity,
│                                        context_based_disparity
├── decoder_only/
│   ├── intrinsic_bias/
│   │   ├── attention_head_based_disparity/      gbe, nie
│   │   └── stereotypical_association/           ca, sll
│   └── extrinsic_bias/
│       ├── counterfactual_fairness/             cr, ctf
│       ├── demographic_representation/          dnp, drd
│       └── performance_disparity/               ad, ba, sns
└── encoder_decoder/
    ├── intrinsic_bias/
    │   ├── algorithmic_disparity/               lfp, mcd
    │   └── stereotypical_association/           sd, sva
    └── extrinsic_bias/                  counterfactual_fairness,
                                         fair_inference, individual_fairness,
                                         position_based
```

This mirrors the organisation of the accompanying textbook, which is why the
directory names are conceptual rather than technical.

## The two axes

**Architecture** decides what a metric can read. An encoder-only model has no
next-token distribution; a decoder-only model has no `[MASK]` position; only an
encoder-decoder generates a translation. This is the same distinction the
`required_task` declaration enforces at runtime. See [Models](api/models.md).

**Intrinsic vs extrinsic** decides what the number means:

| | Reads | Answers | Example |
|---|---|---|---|
| **Intrinsic** | the model's own probabilities or representations | is the bias *in the model* | `weat`, `crows_pairs_score` |
| **Extrinsic** | the outputs of a downstream task | does the bias *cause harm on a task* | `equal_opportunity_gap`, `accuracy_disparity` |

The split matters because the two do not reliably agree. An intrinsic score can
move without any change in task outcomes, and a model can produce equal task
accuracy across groups while encoding the attribute strongly. Reporting one as
evidence for the other is the most common misreading these metrics invite.

| | Encoder-only | Decoder-only | Encoder-decoder |
|---|---|---|---|
| **Intrinsic** | 11 | 4 | 4 |
| **Extrinsic** | 3 | 7 | 4 |

Each cell has a walkthrough: [intrinsic × encoder-only](guides/intrinsic-encoder.md),
[intrinsic × decoder-only](guides/intrinsic-decoder.md),
[intrinsic × encoder-decoder](guides/intrinsic-encdec.md),
[extrinsic × encoder-only](guides/extrinsic-encoder.md),
[extrinsic × decoder-only](guides/extrinsic-decoder.md),
[extrinsic × encoder-decoder](guides/extrinsic-encdec.md).

## Why both layers share one package

Keeping the public and low-level layers under `definitions/` provides three
things:

1. **A stable public surface.** Internals can be reorganised, optimised or
   corrected without changing the root-level classes users import.
2. **One canonical implementation per published metric**, findable by its place
   in the taxonomy rather than by guessing a module name.
3. **Runnable references.** Every leaf carries a `main.py` demonstrating the
   public API on real data, plus its own README:

   ```bash
   python -m fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.main
   ```

!!! warning "Leaf runners are repo-only"
    Some leaves read data files that sit beside them and are **not** shipped in
    the wheel; only the small CrowS-Pairs, WinoBias and Winogender snapshots
    under `fairLMs/datasets/resources/` are packaged. The
    performance-disparity and attention-head leaves in particular need a source
    checkout, and several encoder-decoder leaves download XSum, Europarl or
    XNLI at runtime. The public metrics in `fairLMs.definitions` have no
    such dependency.

## Where the diagnostics sit

`fairLMs/datasets/diagnostics/` is deliberately outside the model-definition taxonomy. It audits datasets
and score tables rather than models, so neither axis applies: there is no
architecture, and the intrinsic/extrinsic distinction is about models, not
evidence. See [Auditing a dataset](guides/dataset-audit.md).
