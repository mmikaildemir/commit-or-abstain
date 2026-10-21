# CaseHOLD Study: Model Prompts and Instructions

As of 2026-10-21.

## Overview

Every model, decision model or generative LLM, receives the same citing passage, the same candidate holdings and the same one-sentence instruction. Only the API shape differs:
- **System One decision models** answer a typed Choice question through the Decisions API.
- **Generative LLMs** answer the same question as a chat message with a forced JSON answer.

The shared instruction, verbatim:

> A court opinion cites a case and the parenthetical describing that case's holding has been replaced by &lt;HOLDING&gt;. Which holding statement is the one the citing court gave for that case?

| Condition | Options shown | Correct answer | Items (full test split) |
| --- | --- | --- | --- |
| Original | A–E: the 5 CaseHOLD holdings | The holding the citing court wrote | 5,314 |
| Abstain | A–E plus NONE | NONE in 1,670 items (31%) whose true holding was swapped out; otherwise the original holding | 5,314 |

In the abstain condition, the swapped-in holding is the real CaseHOLD holding most similar to the removed one by TF-IDF, kept below the 0.75 similarity cap Zheng et al. (2021) used for their distractors. Mean similarity is 0.283, against 0.30 for CaseHOLD's own distractors.

## Task input

Each CaseHOLD row supplies two things the models see:
- `citing_prompt`: an excerpt of a US opinion, with the cited case's parenthetical replaced by `<HOLDING>`.
- `holding_0`–`holding_4`: mapped to option keys A–E in their original order.

The `label` field is never shown, and nothing else is added (no case name of the cited opinion, court or year).

Real example, test row 0 (correct answer C):

```text
Citing passage:
…Some jurisdictions outside of the State of Texas have held that requiring a person to actually go to the location of the sweepstakes sponsor in order to participate constitutes consideration. See Lucky Calendar Co. v. Cohen, 19 N.J. 399, 117 A.2d 487, 496 (1955)(<HOLDING>); Knox Indus. Corp. v. State ex rel. Scanland,

Options:
A. holding a store vicariously liable for wrongful death when its employee shot and killed a customer
B. holding that murder committed by customer was not foreseeable result of excessive sale of alcohol to customer
C. holding that consideration exists where a customer is burdened by having to visit the store where the coupons are being offered
D. holding that negligence of a store customer in failing to perceive a strip of black substance was a question for the jury
E. holding that grocery store did not have duty to foresee and protect customer from rowdy children
```

In the abstain condition the same block gains a sixth option: `NONE. None of the listed holdings is the one the cited case is being cited for in this passage.`

## System One decision models

Decision models get one Choice question per item through `POST https://openrouter.ai/api/alpha/decisions`, with no system prompt, examples or extra settings. The passage goes in `state`, and each option key maps to its holding text in `criteria`. The request is identical for all four decision models except `model`.

```json
{
  "model": "typesafe/jev-1.14",
  "state": {"citing_passage": "<citing_prompt>"},
  "questions": {
    "holding": {
      "type": "choice",
      "instructions": "A court opinion cites a case and the parenthetical describing that case's holding has been replaced by <HOLDING>. Which holding statement is the one the citing court gave for that case?",
      "criteria": {
        "A": "<holding_0>",
        "B": "<holding_1>",
        "C": "<holding_2>",
        "D": "<holding_3>",
        "E": "<holding_4>",
        "NONE": "None of the listed holdings is the one the cited case is being cited for in this passage."
      }
    }
  }
}
```

The `NONE` criterion is sent only in the abstain condition. From each answer we record:
- `choice`: the prediction
- `probabilities`: one per option, used for calibration, thresholds and AUROC
- `confidence`
- `usage.cost`: input tokens only, since output tokens are free

## Generative LLMs

LLMs get one user message per item, with no system prompt and no examples, and must return `{"answer": <key>}`. No confidence is requested, so the prompt matches how a lawyer would actually ask a tool. The message, verbatim with placeholders:

```text
A court opinion cites a case and the parenthetical describing that case's holding has been replaced by <HOLDING>. Which holding statement is the one the citing court gave for that case?

Citing passage:
<citing_prompt>

Options:
A. <holding_0>
B. <holding_1>
C. <holding_2>
D. <holding_3>
E. <holding_4>
NONE. None of the listed holdings is the one the cited case is being cited for in this passage.   <- abstain condition only

Reply with only the option key (A, B, C, D, E, NONE).
```

In the original condition the last line lists only A–E.

Request settings, the same for every LLM:

- **Structured output:** a strict JSON schema whose only field, `answer`, is an enum of the option keys. Without it, some models (Claude Sonnet 5.5) wrote a paragraph before answering.
- **Reasoning:** `effort: "low"`. Vendor defaults ranged from about 200 to 18,000 reasoning tokens per item, and several models treat "low" loosely. So actual reasoning tokens and dollars are recorded per item and reported.
- **Temperature:** 0, where the model accepts it.
- **Logprobs:** top 10 requested. When a model returns them (some open models do), the probability of the answer key is kept as a natural confidence signal. Models that reject the parameter are re-sent without it.

**Sync vs batch:**
- Cheap and mid-tier LLMs run through `POST /api/v1/chat/completions`.
- Frontier models run through OpenRouter's Batch API (`POST /api/v1/batches`). It costs about 50% of the sync price, with results within 24h (Google up to 96h). Requests use the identical message and schema, built by the same function (`run.answer_body`). Batch requests can't carry logprobs.

## Pilot baseline: elicited confidence (appendix only)

In the 1,000-item pilot, GPT-6 Luna was asked for a verbal confidence. This prompt isn't used in the main runs. It's the same message as above, except the last line reads:

```text
Reply with only JSON: {"answer": "<option key>", "confidence": <0-100, your probability that the answer is correct>}
```

That run used temperature 0, the vendor's default reasoning, and no structured output.
- When the true holding was absent, Luna's average confidence on wrong answers was 91% (calibration error 0.29).
- It stays in the appendix as the "LLM self-reported confidence" baseline, because stated confidence isn't something LLMs produce in normal use.

## Models and settings

Ten models, each run on both conditions over all 5,314 test items. Generative LLMs use temperature 0, structured JSON output, `max_tokens` 4096, and the lowest reasoning level each model offers above none.

| Type | Model (OpenRouter id) | API | Run mode | Reasoning setting |
| --- | --- | --- | --- | --- |
| System One | `typesafe/jev-1.14` | Decisions | Sync | n/a |
| System One | `perplexity/pplx-decider-v1.2` | Decisions | Sync | n/a |
| System One | `liquid/d1-ultra` | Decisions | Sync | n/a |
| System One | `openai/gpt-6-luna-decisions` | Decisions | Sync | n/a |
| Generative LLM | `google/gemini-3.8-flash` | Chat | Batch | effort low |
| Generative LLM | `anthropic/claude-haiku-5.5` | Chat | Batch | effort low |
| Generative LLM | `openai/gpt-6-luna` | Chat | Batch | effort low |
| Generative LLM | `mistralai/mistral-small-2603` | Chat | Batch | effort high (only option above none) |
| Generative LLM | `z-ai/glm-5.3-flash` | Chat | Sync | effort low |
| Generative LLM | `openai/gpt-4o-mini` | Chat | Sync | none (no reasoning control) |

One abstain-condition item (id 50733) was refused by Claude Haiku 5.5's provider content filter and is listed in `data/moderation_excluded.json`.

## Design notes and deviations

The setup differs from earlier CaseHOLD evaluations in four deliberate ways:

- **Zero-shot, with no chain of thought in the prompt.** Arvin (ICAIL 2025) asked for written reasoning before `ANSWER: X`. Here, any reasoning is the model's own hidden reasoning, capped at "low" effort, so the two kinds of model differ only in the model, not the prompt.
- **Accuracy plus calibration, not macro F1 alone.** Macro F1 is added in the analysis for comparison with Arvin and LexGLUE.
- **A sixth NONE option in the abstain condition.** CaseHOLD itself has no abstention; this condition is ours.
- **Memorization isn't controlled.** CaseHOLD passages are public US opinions from before 2021, and the dataset has been public since 2021.
  - In testing, Claude Sonnet 5.5 named the cited case and its holding unprompted.
  - pplx-decider reached 88.1%, while the best published results are about 75%.
  - So the study is framed as choosing the right authority versus abstaining, not as a test of legal reasoning. An anonymization check on a subset is planned.

Open question: whether to add Arvin's citation-anonymization test for the decision models, and on how many items.
