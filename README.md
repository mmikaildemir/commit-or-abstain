# Commit or Abstain?

Code, data, and model outputs for **"Commit or Abstain? System One Decision Models, LLMs, and Missing Legal Authority"** (M. Mikail Demir, 2026).

We extend [CaseHOLD](https://github.com/reglab/casehold) with an *abstain* condition. In 1,670 of the 5,314 test items, the correct holding is replaced by a similar real holding from another case, and a sixth option states that none of the holdings applies. We compare four System One decision models with six generative LLMs on both conditions. We also test policies that combine the two kinds of model, check for contamination, and distill a decision model's probabilities into Qwen3-8B.

Every number in the paper can be recomputed from the files in this repository without calling any API.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/full.jsonl` | All 5,314 CaseHOLD test items, with both conditions (`original` and `abstain`) per item |
| `data/sample.jsonl` | The 1,000-item pilot sample (Appendix B) |
| `data/contam*.jsonl` | The 406 contamination-check items and their three variants (Appendix C) |
| `data/moderation_excluded.json` | The one item refused by a provider's content filter |
| `logs_full/` | Per-item outputs of all ten models on both conditions, plus the Qwen3-8B and BERT distillation runs |
| `logs_contam_*/` | Per-item outputs on the contamination variants |
| `logs_train/` | Jev's probabilities on 10,000 training items, used as distillation targets |
| `logs/` | Pilot runs, including stated confidence (Appendix B) |
| `final_stats.json`, `combine_full.json`, `contam_table.json` | The computed results behind the paper's tables |
| `figures/` | Scripts that draw Figures 1 to 3 from the logs |
| `PROMPTS.md` | The exact instructions, request formats, and model settings |

Each log row holds the item id, the gold answer, the model's answer, its probability for each option (decision models and local models), the billed cost, and the raw API response.

## Reproducing the results from the saved outputs

```bash
pip install -r requirements.txt
CASEHOLD_SET=full python final_stats.py      # Table 2: accuracy, wrong picks, false abstention, AUROC, ECE, cost, CIs
CASEHOLD_SET=full python combine_full.py     # Section 4.3: combination policies on the held-out 70%
python figures/make_fig1_scatter.py          # Figure 1 (wrong picks against cost)
python figures/make_fig2_roc.py              # Figure 2 (ROC)
python figures/make_fig3_tradeoff.py         # Figure 3 (trade-off)
```

The scripts read from `logs_full/` and write their output next to themselves.

## Re-running the models

Model calls go through [OpenRouter](https://openrouter.ai). Put your key in a `.env` file (see `.env.example`) or export `OPENROUTER_API_KEY`.

```bash
# Build the two conditions from the CaseHOLD test split (downloads rows through the Hugging Face API)
python fetch.py all

# Decision models (Decisions API)
CASEHOLD_SET=full python run.py typesafe/jev-1.14 abstain

# Generative LLMs, answer only, synchronous
CASEHOLD_SET=full python run.py ans:openai/gpt-4o-mini abstain

# Generative LLMs through the Batch API (about half the price)
CASEHOLD_SET=full python batch.py submit google/gemini-3.8-flash abstain
CASEHOLD_SET=full python batch.py collect
```

Runs resume where they stopped, since items already in a log file are skipped. Hosted models change over time, so new runs may not match the saved outputs exactly. The paper reports versions available in October 2026.

**Contamination checks.** `python contam.py build` samples 500 items and rewrites them with GPT-6 Luna. After validation, 406 items remain. Run a model on a variant with `CASEHOLD_SET=contam_anon python run.py MODEL original`.

**Distillation** needs a GPU (we used one NVIDIA RTX A6000) and the CaseHOLD training split:

```bash
python prep_train.py data/train.parquet data/train.jsonl   # train.parquet from the casehold/casehold dataset on Hugging Face
python llm_distill.py --run L0 --cond abstain                                                    # Qwen3-8B, no fine-tuning
python llm_distill.py --run L1 --cond abstain --teacher logs_train/typesafe_jev-1.14__abstain.jsonl  # Jev's probabilities only
python llm_distill.py --run L2 --cond abstain --gold                                             # gold labels
```

`distill.py` runs the BERT variant reported in Appendix B.

## Data and license

The code is released under the MIT License. The items in `data/` are derived from CaseHOLD (Zheng et al., ICAIL 2021) and remain subject to its license. Please cite CaseHOLD if you use them.

## Citation

```bibtex
@misc{demir2026commit,
  author = {Demir, M. Mikail},
  title  = {Commit or Abstain? {System One} Decision Models, {LLMs}, and Missing Legal Authority},
  year   = {2026}
}
```
