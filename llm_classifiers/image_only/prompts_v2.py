#!/usr/bin/env python3
"""Second-round prompt strategies for image-only knee MRI classification.

The first sweep established that the failure is a decoding/prompting one, not a lack
of pixels: every rendering variant hedged identically, and what broke the hedge was
making the model commit to something before scoring. These strategies push on that.

Five directions:

  background   report-derived context on what a knee MRI reader actually looks at
  questions    a fixed structured read-out answered before any score is given
  verbal       coarse word scales, and scores read from the answer-token distribution
  averaging    several seeded samples at nonzero temperature, averaged
  few-shot     labelled example studies for the finding being predicted

The background block is derived from the 4,349 *unlabeled* training reports only. The
58 labelled studies are never used to build a prompt, so nothing here leaks the
outcome being scored. Few-shot examples do carry labels, but only from studies other
than the one under test — see `select_examples` in the runner.
"""

from __future__ import annotations

from dicom_io import LABELS
from prompts import DEFINITIONS


# ---------------------------------------------------------------------------
# Direction 1: background knowledge and a structured read-out
# ---------------------------------------------------------------------------

# Frequencies measured over 1,715 English-language unlabeled training reports. They
# describe what radiologists writing about these scans habitually comment on, which
# is a reasonable proxy for what is visible and worth checking. This is background,
# not instruction: it says what a reader attends to, not what to conclude.
BACKGROUND = """Background on how these knee MRI studies are normally read.

Radiologists reporting this collection work through a fixed search pattern, and
comment on the following structures in roughly this order and this frequency:

- joint fluid / effusion (90% of reports), cartilage and chondral surfaces (89%)
- anterior cruciate ligament (88%), posterior cruciate ligament (83%)
- medial collateral ligament (81%), lateral collateral ligament (72%)
- lateral meniscus (77%), medial meniscus (73%)
- extensor mechanism (75%), patellofemoral compartment (67%)
- bone marrow oedema or bruising (52%), popliteal/Baker's cyst (51%)
- fracture (35%), Hoffa's fat pad (31%), synovium (21%)

Useful regularities in how findings present together:

- Effusion is graded trace / small / moderate / large by how far fluid distends the
  suprapatellar recess. It is the most commonly reported positive finding.
- Synovitis is rarely reported in isolation. It is almost always described alongside
  an effusion, as "synovial proliferation" or "thickened synovial tissue". An
  effusion with irregular, thickened or frond-like soft tissue inside it is the
  usual appearance; a clear, uniform effusion is not synovitis.
- A meniscal tear is called when signal reaches an articular surface. Linear signal
  in the posterior horn extending to the surface is the commonest description.
  Intrasubstance signal that does not reach a surface is explicitly not a tear.
- ACL tears are graded partial or complete and are often accompanied by bone
  contusions in a pivot-shift pattern: lateral femoral condyle and posterolateral
  tibial plateau. Anterior tibial translation is a secondary sign.
- Cartilage loss is graded by depth (low-grade under 50%, high-grade over 50%,
  full-thickness) and located by compartment: medial, lateral, patellofemoral.
- Fractures in this collection are usually subtle: a thin low-signal line, a
  subchondral or insufficiency fracture, often with surrounding marrow oedema
  rather than displacement.
- Baker's cysts arise between the medial head of gastrocnemius and semimembranosus
  and are frequently seen together with an effusion."""


# One question per structure, in the order the reports themselves follow. The model
# answers all of them before it is allowed to score anything, so that the score is
# conditioned on stated observations rather than produced from a blank page.
READ_QUESTIONS = """Answer each question from the images. One short sentence each.
If a structure is not adequately shown on these slices, say so rather than guessing.

1. Is there fluid distending the suprapatellar recess? If so, is it trace, small,
   moderate or large?
2. Within any effusion, is the synovium smooth and thin, or thickened, irregular or
   frond-like? Is Hoffa's fat pad of normal signal?
3. Is the anterior cruciate ligament continuous, with taut parallel low-signal
   fibres, or is it thickened, bright, wavy or absent?
4. Is the posterior cruciate ligament intact?
5. On coronal images, is the superficial medial collateral ligament of normal
   thickness and signal, or is it thickened with surrounding fluid?
6. Medial meniscus: is there linear or complex signal reaching a superior or
   inferior articular surface, truncation, or a displaced fragment?
7. Lateral meniscus: the same question.
8. Medial compartment: is the femoral and tibial cartilage of full thickness? Are
   there osteophytes or subchondral cysts or sclerosis?
9. Lateral compartment: the same question.
10. Patellofemoral compartment: is the patellar and trochlear cartilage intact, or
    fissured, thinned or ulcerated?
11. Bone marrow: is there ill-defined high signal on fluid-sensitive fat-suppressed
    images? If so, in which bone and does it sit under a joint surface?
12. Is there any cortical break or low-signal fracture line?
13. Is there a fluid collection in the popliteal fossa between the medial head of
    gastrocnemius and semimembranosus?"""


def background_prefix(use_background: bool) -> str:
    return f"{BACKGROUND}\n\n" if use_background else ""


# ---------------------------------------------------------------------------
# Direction 2: verbal confidence scales
# ---------------------------------------------------------------------------

# The scale the user proposed: five levels with an explicit "unclear" midpoint, which
# gives the model somewhere honest to put a finding it genuinely cannot resolve on
# sparse slices, instead of forcing it toward "absent".
WORDS5 = ["very unlikely", "unlikely", "unclear", "likely", "very likely"]
WORDS5_VALUES = dict(zip(WORDS5, [0.05, 0.25, 0.50, 0.75, 0.95]))

# Single-token, mutually distinct first tokens, so the whole distribution over the
# scale can be read from one generated token's logprobs. "very unlikely" and "very
# likely" share a first token and could not be separated this way.
DIGITS = ["0", "1", "2", "3", "4", "5"]
DIGIT_VALUES = {d: i / 5.0 for i, d in enumerate(DIGITS)}

# A word scale that is also readable from one token. WORDS5 cannot be: "very
# unlikely" and "very likely" share their first token and would be indistinguishable.
# These five are mutually distinct at the first token while keeping the same ordinal
# meaning, including an explicit middle for "the slices cannot resolve this".
WORDS_LP = ["absent", "improbable", "unclear", "probable", "certain"]
WORDS_LP_VALUES = dict(zip(WORDS_LP, [0.05, 0.25, 0.50, 0.75, 0.95]))

DIGIT_SCALE = """Rate how confident you are that this finding is present in this knee:
0 = definitely absent
1 = probably absent
2 = leaning absent, but unsure
3 = leaning present, but unsure
4 = probably present
5 = definitely present"""


def digit_scale_prompt(captions: list[str], label: str, use_background: bool) -> str:
    from prompts import _image_manifest

    return (
        f"{background_prefix(use_background)}{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study.\n\n"
        f"Finding: {label} - {DEFINITIONS[label]}\n\n"
        f"{DIGIT_SCALE}\n\nAnswer with exactly one digit, 0 to 5, and nothing else."
    )


def words_prompt(captions: list[str], label: str, use_background: bool) -> str:
    from prompts import _image_manifest

    return (
        f"{background_prefix(use_background)}{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study.\n\n"
        f"Finding: {label} - {DEFINITIONS[label]}\n\n"
        "How likely is it that this finding is present in this knee? Answer with "
        "exactly one of: " + ", ".join(f'"{w}"' for w in WORDS_LP) + ", and nothing "
        'else. Use "unclear" only when these slices genuinely cannot resolve it.'
    )


# ---------------------------------------------------------------------------
# Direction 1 continued: two-stage with a structured read-out
# ---------------------------------------------------------------------------

QUESTIONS_SYSTEM = (
    "You are a musculoskeletal radiologist reading a knee MRI. Answer the numbered "
    "questions from the images only. Do not give probabilities and do not summarise."
)


def questions_prompt(captions: list[str], use_background: bool) -> str:
    from prompts import _image_manifest

    return (
        f"{background_prefix(use_background)}{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study.\n\n"
        f"{READ_QUESTIONS}"
    )


def score_answers_prompt(answers: str, scale: str) -> str:
    body = (
        "A radiologist worked through a structured read-out of a knee MRI from a "
        "sparse subset of slices and recorded these answers. The read-out may be "
        "incomplete or mistaken, and a structure it calls normal may simply not have "
        "been well shown.\n\n"
        f"---\n{answers}\n---\n\n"
        f"Findings to score:\n\n"
        + "\n".join(f"- {label}: {DEFINITIONS[label]}" for label in LABELS)
        + "\n\n"
    )
    if scale == "words5":
        return (
            body
            + "For each finding, say how likely it is to be present in this knee using "
            "exactly one of: " + ", ".join(f'"{w}"' for w in WORDS5) + ". "
            "Return JSON only, with one key per finding, using exactly the names above."
        )
    return (
        body + "Give the probability from 0 to 1 that each finding is present in this "
        "knee. Return JSON only, with one key per finding, using exactly the names above."
    )


# ---------------------------------------------------------------------------
# Direction 5: few-shot
# ---------------------------------------------------------------------------

FEWSHOT_SYSTEM = (
    "You are a musculoskeletal radiologist. You will be shown labelled example knee "
    "MRI studies for one specific finding, then an unlabelled study to judge. The "
    "examples are other patients; use them to calibrate what the finding looks like "
    "on this kind of scan, not as evidence about the final case."
)


def fewshot_intro(label: str, n_each: int, use_background: bool) -> str:
    return (
        f"{background_prefix(use_background)}"
        f"Finding to judge: {label} - {DEFINITIONS[label]}\n\n"
        f"First, {2 * n_each} labelled example studies from other patients "
        f"({n_each} where this finding is present, {n_each} where it is absent), "
        "shown in a random order. Then the study you must judge."
    )


def fewshot_example_caption(index: int, present: bool, n_images: int) -> str:
    verdict = "PRESENT" if present else "ABSENT"
    return (
        f"Example {index} ({n_images} slices) - this finding is {verdict} in this study."
    )


def fewshot_question(label: str, scale: str) -> str:
    lead = (
        "Those were the labelled examples. The images above this line are the study to "
        f"judge. Considering how {label} appeared in the examples, judge this study.\n\n"
    )
    if scale == "digit":
        return lead + DIGIT_SCALE + "\n\nAnswer with exactly one digit, 0 to 5, and nothing else."
    if scale == "words5":
        return (
            lead + "Answer with exactly one of: "
            + ", ".join(f'"{w}"' for w in WORDS5)
            + "."
        )
    return lead + "Is this finding present in this knee? Answer with exactly one word: Yes or No."
