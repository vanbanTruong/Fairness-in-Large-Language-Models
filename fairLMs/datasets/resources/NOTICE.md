# Third-party data and provenance

FairLMs software uses the repository MIT license. Third-party datasets retain
separate licenses; they are not relicensed by installation of the package.
This notice was checked against the linked sources on 2026-09-24.

| Dataset | Creators / source | Terms reported by source | Delivery and snapshot |
|---|---|---|---|
| CrowS-Pairs | Nangia, Vania, Bhalerao & Bowman (2020), [NYU dataset card](https://huggingface.co/datasets/nyu-mll/crows_pairs) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Bundled CSV; exact bytes in checksums.json |
| BBQ | Parrish et al. (2022), [NYU repository](https://github.com/nyu-mll/BBQ) | [CC BY 4.0](https://github.com/nyu-mll/BBQ/blob/main/LICENSE) | Downloaded from the Hugging Face mirror and cached at first use |
| StereoSet | Nadeem, Bethke & Reddy (2021), [McGill card](https://huggingface.co/datasets/McGill-NLP/stereoset/blob/main/README.md) | CC BY-SA 4.0, as linked above | Downloaded; pin `revision=` for a particular run |
| WinoBias | Zhao et al. (2018), [UCLA repository](https://github.com/uclanlp/corefBias) and [dataset card](https://huggingface.co/datasets/uclanlp/wino_bias) | MIT; license text included | Eight bundled Parquet splits from Hub revision `3f31267586e4408e3b3f77ec22198fd24ea8dc1d`; official license from repository revision `0bce984dd081bbc10b0622f326727a024c607895` |
| Winogender | Rudinger et al. (2018), [official repository](https://github.com/rudinger/winogender-schemas) | MIT; license text included | Bundled sentences, templates and occupation statistics from revision `1c7f8b481ad8a234b41e9f76a424d6e856e13f7f` |
| Bias in Bios | De-Arteaga et al. (2019), [LabHC card](https://huggingface.co/datasets/LabHC/bias_in_bios/blob/main/README.md) | MIT, per card | Downloaded LabHC version; differs from original biography collection |
| XNLI | Conneau et al. (2018), [official repository](https://github.com/facebookresearch/XNLI) | [CC BY-NC 4.0](https://github.com/facebookresearch/XNLI/blob/main/LICENSE) | Downloaded; swapped premises are derived counterfactual evidence |

The bundled CrowS-Pairs file is inherited unchanged from the supplied FairLMs
snapshot, associated with commit `7258ac2c405475998269257ff41271ef6cfadf8e`.
Their original upstream dataset commit is not recorded in that snapshot.
The manifest identifies actual bundled bytes; it does not invent an upstream
revision. Byte-identical legacy copies were removed, without changing canonical
corpus contents. WinoBias and Winogender were downloaded from the revisions
listed above. `checksums.json` pins every bundled dataset file.

StereoSet transformations now preserve roles and contexts. XNLI transformations
use word-boundary substitutions and remove duplicate reverse pairs; they do not
create human stereotype labels. Returned rows identify the pairing source.

Bundled WEAT and SEAT word stimuli retain their references to Caliskan et al.
(2017) and May et al. (2019) in `fairLMs/definitions/resources/word_sets.py` and the original
`definitions/` modules. This snapshot does not establish a separate license for
those term-list compilations or the legacy BiasAsker CSV files: upstream
attribution/version confirmation for those inherited assets remains open.
Historical MCD outputs from earlier development are not distributed with the
package or kept in this repository; they are not validation results for this
version.
