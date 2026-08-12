#!/usr/bin/env python3
"""Zero-shot prompt strategies for image-only knee MRI classification.

Every strategy sees only pixel data plus the acquisition descriptors already
published in `train_series.csv` (plane, fluid sensitivity, fat suppression).
No report text, no labels, and no worked examples are ever supplied, so all
strategies here are genuinely zero-shot.
"""

from __future__ import annotations

from dicom_io import LABELS


# Imaging-side definitions: what the finding looks like on knee MRI, and the usual
# confusers. Written to be readable by the model without presuming a label prior.
DEFINITIONS = {
    "ACL": (
        "Anterior cruciate ligament injury: discontinuity, non-visualisation, abnormal "
        "high signal or wavy/horizontal course of the ACL, best seen on sagittal images."
    ),
    "MCL": (
        "Medial collateral ligament injury: thickening, high signal, or fluid tracking "
        "along the superficial medial collateral ligament on coronal images. Isolated "
        "subcutaneous medial soft-tissue oedema is not sufficient."
    ),
    "Medial Meniscus": (
        "Medial meniscus tear: signal reaching an articular surface of the medial "
        "meniscus, abnormal shape, truncation, displaced fragment, root tear, or "
        "maceration. Intrasubstance degeneration that does not reach a surface does not count."
    ),
    "Lateral Meniscus": (
        "Lateral meniscus tear: the same surfacing-signal, shape and displacement criteria "
        "applied to the lateral meniscus."
    ),
    "Medial OA": (
        "Medial tibiofemoral osteoarthritis: cartilage thinning or full-thickness loss, "
        "subchondral sclerosis/cysts, and osteophytes in the medial compartment."
    ),
    "Lateral OA": (
        "Lateral tibiofemoral osteoarthritis: the same degenerative changes in the lateral "
        "compartment."
    ),
    "PF OA": (
        "Patellofemoral osteoarthritis: patellar or trochlear cartilage fissuring, thinning, "
        "ulceration or full-thickness defect, with subchondral change or osteophytes."
    ),
    "Effusion": (
        "Joint effusion: abnormal fluid signal distending the joint capsule and "
        "suprapatellar recess on fluid-sensitive images, including small effusions."
    ),
    "Synovitis": (
        "Synovitis: thickened, irregular or proliferative synovium, often with intermediate "
        "signal within the effusion or Hoffa's fat-pad oedema. Simple effusion alone is not "
        "sufficient."
    ),
    "Baker's": (
        "Baker's (popliteal) cyst: fluid collection with a neck between the medial head of "
        "gastrocnemius and semimembranosus, best seen on axial images."
    ),
    "Contusion": (
        "Bone contusion / bone marrow oedema of traumatic pattern: ill-defined marrow high "
        "signal on fluid-sensitive fat-suppressed images, not attributable solely to "
        "degenerative subchondral change."
    ),
    "Fracture": (
        "Fracture: a low-signal fracture line, cortical break, impaction, or bony avulsion, "
        "including subchondral insufficiency and stress fractures."
    ),
}


SYSTEM_BASE = (
    "You are a musculoskeletal radiologist reading a knee MRI. You are shown a sparse "
    "subset of slices from several sequences of one study. Judge only what the images "
    "support. Because the slices are sparse, absence of a visible finding is weaker "
    "evidence than presence of one, and your probabilities should reflect that."
)

SYSTEM_TERSE = (
    "You are a radiologist reading a knee MRI. Answer only with what is requested."
)


def _definition_block(labels: list[str]) -> str:
    return "\n".join(f"- {label}: {DEFINITIONS[label]}" for label in labels)


def _image_manifest(captions: list[str]) -> str:
    lines = [f"Image {i}: {caption}" for i, caption in enumerate(captions, start=1)]
    return "The images, in order, are:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Joint strategies: all twelve findings in one request
# ---------------------------------------------------------------------------


def joint_plain(captions: list[str]) -> str:
    return (
        f"{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study. For each of the following "
        "findings, give the probability from 0 to 1 that the finding is present in this "
        "knee.\n\n" + "\n".join(f"- {label}" for label in LABELS) + "\n\n"
        "Return JSON only, with one key per finding, using exactly these names."
    )


def joint_definitions(captions: list[str]) -> str:
    return (
        f"{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study. Findings to assess:\n\n"
        f"{_definition_block(LABELS)}\n\n"
        "For each finding give the probability from 0 to 1 that it is present in this "
        "knee. Use the full range: 0.5 means genuinely uncertain, and values near 0 or 1 "
        "should be reserved for findings you can clearly exclude or confirm on these "
        "slices. Return JSON only, with one key per finding, using exactly the names above."
    )


def joint_checklist(captions: list[str]) -> str:
    return (
        f"{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study. Work through a systematic search "
        "pattern before answering:\n"
        "1. Fluid: suprapatellar recess distension, and any popliteal fluid collection with a neck.\n"
        "2. Synovium: thickness, irregularity, and Hoffa's fat-pad signal.\n"
        "3. Cruciates: ACL continuity, fibre orientation and signal on sagittal images.\n"
        "4. Collaterals: superficial MCL thickness, signal and surrounding fluid on coronal images.\n"
        "5. Menisci: each horn of the medial and lateral meniscus for surfacing signal, "
        "truncation or displacement.\n"
        "6. Cartilage and bone: medial, lateral and patellofemoral compartments for cartilage "
        "loss, osteophytes and subchondral change.\n"
        "7. Marrow: traumatic-pattern oedema, fracture lines, cortical breaks.\n\n"
        f"Findings to score:\n\n{_definition_block(LABELS)}\n\n"
        "Then give the probability from 0 to 1 for each finding. Return JSON only, with "
        "one key per finding, using exactly the names above."
    )


def joint_prevalence_free(captions: list[str]) -> str:
    """Same as `joint_definitions` but explicitly discourages a uniform hedge.

    Guided decoding on a 4B model tends to emit a constant probability for every key;
    this variant tests whether an explicit anti-degeneracy instruction fixes it.
    """
    return (
        f"{joint_definitions(captions)}\n\n"
        "Do not give the same probability to every finding. Rank the findings against each "
        "other: the finding you consider most likely in this knee must receive a strictly "
        "higher probability than the one you consider least likely."
    )


# ---------------------------------------------------------------------------
# Per-finding strategies
# ---------------------------------------------------------------------------


def binary_json(captions: list[str], label: str) -> str:
    return (
        f"{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study.\n\n"
        f"Finding: {label} - {DEFINITIONS[label]}\n\n"
        "Give the probability from 0 to 1 that this finding is present in this knee. "
        "Return JSON only."
    )


def binary_yesno(captions: list[str], label: str) -> str:
    """Single-token question; the score is read from the yes/no logprobs."""
    return (
        f"{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study.\n\n"
        f"Finding: {label} - {DEFINITIONS[label]}\n\n"
        "Is this finding present in this knee? Answer with exactly one word: Yes or No."
    )


# ---------------------------------------------------------------------------
# Two-stage: free-text reading, then text-only scoring of that reading
# ---------------------------------------------------------------------------

DESCRIBE_SYSTEM = (
    "You are a musculoskeletal radiologist. Describe only what you can see in the supplied "
    "knee MRI slices. Do not speculate beyond the images and do not give probabilities."
)


def describe_prompt(captions: list[str]) -> str:
    return (
        f"{_image_manifest(captions)}\n\n"
        "These images all come from one knee MRI study. Write a concise structured reading "
        "covering, in order: joint fluid, synovium and fat pad, cruciate ligaments, "
        "collateral ligaments, medial meniscus, lateral meniscus, cartilage in each of the "
        "three compartments, bone marrow signal and cortical integrity, and the popliteal "
        "fossa. One or two sentences per heading. State explicitly when a structure is not "
        "adequately shown on these slices."
    )


def score_reading_prompt(reading: str) -> str:
    return (
        "A radiologist read a knee MRI and produced the following observations from a sparse "
        "subset of slices. The reading may be incomplete or mistaken, and a structure it "
        "calls normal may simply not have been well shown.\n\n"
        f"---\n{reading}\n---\n\n"
        f"Findings to score:\n\n{_definition_block(LABELS)}\n\n"
        "Give the probability from 0 to 1 that each finding is present in this knee. "
        "Return JSON only, with one key per finding, using exactly the names above."
    )


JOINT_STRATEGIES = {
    "joint_plain": joint_plain,
    "joint_definitions": joint_definitions,
    "joint_checklist": joint_checklist,
    "joint_prevalence_free": joint_prevalence_free,
}

PER_LABEL_STRATEGIES = {
    "binary_json": binary_json,
    "binary_logprob": binary_yesno,
}
