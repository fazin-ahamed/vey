#!/usr/bin/env python3
"""Author and stream the prospective ECA-1 Evidence Pages assay corpus.

`--prepare-source` freezes authored material and emits an opaque review packet
plus a separate sealed key. `--build` is deliberately gated on a complete,
accepted, source-hash-bound independent review receipt. This module never loads
or scores a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "research" / "endgame" / "ephemeral_pages_protocol.json"
CANONICAL_ROOT = Path("/home/fazinahamed/Documents/vey")
if str(CANONICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CANONICAL_ROOT))
from vey_u.ir import Candidate, DecisionIR, Evidence, StateBlock  # noqa: E402


SOURCE_SCHEMA = "vey.eca1.authored-source.v1"
INVENTORY_SCHEMA = "vey.eca1.source-inventory.v1"
PACKET_SCHEMA = "vey.eca1.opaque-review-packet.v1"
KEY_SCHEMA = "vey.eca1.sealed-target-key.v1"
RECEIPT_SCHEMA = "vey.eca1.independent-audit-receipt.v1"
BUILD_SCHEMA = "vey.eca1.corpus-build-manifest.v1"
SEED = 1729
SPLITS = ("train", "validation", "calibration", "development", "final")
WORLD_COUNTS = {"train": 128, "validation": 64, "calibration": 128, "development": 128, "final": 320}
GRADE_ORDER = (0, 1, 2, 3, 4)
UNKNOWN_ID = "__unknown__"
EXPECTED_CANONICAL_FILES = {
    "vey_u/ir.py": "0184e25e05c17638117e776fc660fd6e4896c3590da0fcf94fc753509869ea2d",
    "vey_u/semantic/format.py": "3e7ad63a26b68e7bb74142d8c69306bd4c0473f42eb5a69680e0474422c0e1ab",
    "vey_u/semantic/models.py": "421f8d20aab9f1b545d18ad74eee6cf02c00fd13f0a7feac841e848f366dbc94",
}

FAMILIES = (
    "quality", "cost", "risk", "safety", "urgency", "relevance", "suitability",
    "intent", "compliance", "plausibility", "sentiment", "similarity", "priority",
    "trust", "effort", "latency", "durability", "reliability", "privacy", "business impact",
)

# Each five-clause sequence describes extent of its own named property. Several
# properties intentionally increase undesirable conditions: orientation, not
# desirability, determines whether more or less extent is preferred.
AUTHORED_FIELDS: tuple[tuple[str, str, str, str, str, tuple[str, str, str, str, str]], ...] = (
    ("quality", "A", "Outcome repeatability", "consistency in repeated outcomes", "how closely an operating process reproduces its results across attempts", (
        "repeated runs produce results that differ in their principal details", "familiar runs return a recognizable result while ordinary changes still alter important details", "routine runs reproduce the main result although unfamiliar inputs still shift some details", "routine and unfamiliar runs reproduce the result with changes confined to unusual conditions", "the same result recurs across routine and unusual conditions with its principal details intact")),
    ("quality", "B", "Procedural instruction clarity", "clarity of step-by-step instructions", "how readily a reader can interpret and follow the stated procedure", (
        "the procedure leaves its actions and their order unstated", "the procedure names familiar actions but leaves transitions open to interpretation", "the procedure explains the usual sequence while some transitions require inference", "the procedure makes its sequence and most transitions explicit", "the procedure states each action and transition in language that readers interpret consistently")),
    ("quality", "C", "Recurring output defects", "recurrence of the same output defects", "how persistently the same fault pattern appears in produced results", (
        "reviewers find no recurring fault pattern in the produced results", "an isolated result shows a fault that does not recur in later checks", "the same fault returns in familiar conditions but not across ordinary use", "the same fault returns across ordinary use and recedes only in unusual cases", "the same fault appears across routine and unusual conditions")),

    ("cost", "A", "Initial setup outlay", "upfront setup expenditure", "the amount of resource expenditure required before ordinary operation begins", (
        "the quoted setup requires no separately billed outlay", "the quote folds setup into an existing charge without a distinct payment", "the quote lists a separate setup payment alongside the ordinary charge", "the setup payment occupies a substantial part of the quoted initial spend", "the setup requires a major upfront payment before operation can begin")),
    ("cost", "B", "Recurring operating spend", "recurring operating expenditure", "the resource expenditure that accumulates during continued ordinary operation", (
        "routine operation adds no recurring expenditure", "only occasional activity adds a recurring charge to the account", "ordinary use adds a recurring charge that remains visible on periodic statements", "most operating cycles add a sizeable recurring charge", "continued operation draws a substantial recurring share of available resources")),
    ("cost", "C", "Exit and switching fees", "fees incurred when ending or changing a service", "the financial charge attached to leaving an arrangement or moving to another", (
        "ending the arrangement carries no separate fee", "a fee appears only when a special exit service is requested", "ordinary cancellation triggers a distinct closing charge", "leaving early triggers several charges tied to the remaining arrangement", "switching requires a large settlement before another service can be used")),

    ("risk", "A", "Supplier concentration", "concentration of supply in a single provider", "how much delivery depends on one source rather than independent alternatives", (
        "several independent providers can supply the required input", "one provider supplies a narrow component while alternatives cover the rest", "the primary provider supplies a substantial share and alternatives need adjustment", "the primary provider supplies nearly every critical input", "all critical supply depends on one provider with no ready alternative")),
    ("risk", "B", "Demand forecast volatility", "volatility in future demand", "how much expected demand shifts between planning periods", (
        "successive planning periods show a stable demand pattern", "small revisions follow an otherwise steady pattern", "revisions regularly change the expected demand profile", "large shifts recur between planning periods", "the expected demand pattern changes sharply from one period to the next")),
    ("risk", "C", "Unmitigated dependency exposure", "exposure to dependencies without a fallback", "how much an operation relies on dependencies that lack a workable substitute", (
        "critical dependencies each have a tested substitute", "one noncritical dependency lacks a substitute", "a critical dependency has a substitute that cannot cover ordinary demand", "several critical dependencies lack a workable substitute", "operation stops when a single unbacked dependency is unavailable")),

    ("safety", "A", "Unshielded heat contact", "exposure to unshielded hot surfaces", "how much ordinary work brings people into contact with unshielded heat", (
        "work keeps people separated from hot surfaces", "contact is possible only during a controlled maintenance step", "routine handling passes near an unshielded hot surface", "ordinary work repeatedly places hands beside an unshielded hot surface", "the normal working position requires sustained contact with an unshielded hot surface")),
    ("safety", "B", "Guard interruption", "frequency of interruptions to protective barriers", "how often a protective barrier is opened or bypassed during operation", (
        "the protective barrier remains in place throughout ordinary operation", "the barrier is opened for a limited service task", "the barrier is opened during recurring operating tasks", "the barrier is bypassed in most operating cycles", "ordinary operation proceeds with the barrier routinely bypassed")),
    ("safety", "C", "Spill escape propensity", "likelihood of material escaping spill containment", "how readily a spill passes beyond its intended containment area", (
        "containment retains the material during the described release", "a minor leak stays within the immediate collection area", "a spill can cross the collection area during ordinary handling", "a spill commonly reaches adjacent work surfaces", "a release can spread beyond the work area before containment takes effect")),

    ("urgency", "A", "Response window narrowness", "narrowness of the available response window", "how little time remains between receiving a request and its consequence", (
        "the request remains actionable through an open-ended planning period", "the request can wait through routine scheduling without consequence", "the request needs attention within the current operating cycle", "the request must be handled before the next planned cycle", "the request becomes unusable if action does not begin immediately")),
    ("urgency", "B", "Deadline sensitivity", "sensitivity to a fixed deadline", "how strongly a result depends on completion before a stated cutoff", (
        "completion after the stated date leaves the outcome unchanged", "a modest delay changes convenience but not the intended outcome", "a delay past the expected window requires the plan to be revised", "missing the cutoff removes an important part of the intended outcome", "crossing the cutoff makes the requested outcome unavailable")),
    ("urgency", "C", "Delay-linked downstream harm", "harm associated with response delay", "how much downstream disruption accumulates while a response is deferred", (
        "deferral leaves downstream work unaffected", "a deferral creates a minor follow-up adjustment", "a deferral interrupts a dependent task", "a deferral disrupts several connected tasks", "a deferral causes a consequential loss that cannot be recovered within the workflow")),

    ("relevance", "A", "Subject match", "match to the requested subject", "how closely the described material concerns the subject named by the request", (
        "the material concerns a different subject", "the material mentions the requested subject only in passing", "the material addresses a related aspect of the requested subject", "the material directly discusses the requested subject and its main context", "the material stays centered on the requested subject throughout")),
    ("relevance", "B", "Intended-use alignment", "alignment with the intended use context", "how closely the material fits the setting in which the result will be used", (
        "the material belongs to a setting unrelated to the intended use", "the material shares background context but not the intended use", "the material fits a neighboring use with some adaptation", "the material fits the stated use with only a narrow adjustment", "the material was produced for the same use and operating context")),
    ("relevance", "C", "Evidence directness", "directness of evidence for the stated criterion", "how directly the described evidence supports the question being asked", (
        "the record contains no evidence bearing on the criterion", "the record offers an indirect clue that needs another inference", "the record supports a related part of the criterion", "the record directly supports most of the criterion", "the record directly documents the criterion itself")),

    ("suitability", "A", "Task-constraint compatibility", "compatibility with task constraints", "how well the described option fits the task's required operating conditions", (
        "the option conflicts with a required task condition", "the option meets background conditions but misses a central requirement", "the option meets the central requirement after a workflow adjustment", "the option meets the stated requirements with a narrow accommodation", "the option fits the task requirements in its ordinary operating form")),
    ("suitability", "B", "Workflow fit", "fit with the user's working routine", "how readily the option fits the sequence and tools of the intended workflow", (
        "the option cannot be used within the described working routine", "the option requires replacing a central step in the routine", "the option fits after the user adds a separate handoff", "the option fits the usual sequence with a small local change", "the option follows the user's existing sequence without a separate handoff")),
    ("suitability", "C", "Environmental adaptation", "adaptation to the operating environment", "how well the option continues to function under the environment's stated conditions", (
        "the option fails under a routine environmental condition", "the option operates only in a protected corner of the environment", "the option tolerates routine conditions but not their ordinary variation", "the option adapts to ordinary variation with occasional adjustment", "the option remains suited across the routine and varying conditions described")),

    ("intent", "A", "Requested-action explicitness", "explicitness of the requested action", "how clearly the speaker states the action they want performed", (
        "the speaker raises a topic without stating an action", "the speaker hints at a possible action but leaves the request open", "the speaker names an action while leaving its intended outcome unclear", "the speaker asks for a defined action and its expected result", "the speaker directly specifies the action, outcome, and desired next step")),
    ("intent", "B", "Purchase readiness", "strength of purchase-readiness signals", "how far the expressed behavior has progressed toward completing a purchase", (
        "the speaker is only gathering general information", "the speaker compares options without discussing a transaction", "the speaker asks about terms or availability for a possible purchase", "the speaker requests the steps needed to place an order", "the speaker confirms the selected option and asks to complete the transaction")),
    ("intent", "C", "Cancellation intent", "strength of cancellation intent", "how clearly and firmly a speaker indicates a wish to end an arrangement", (
        "the speaker gives no indication of wanting to end the arrangement", "the speaker asks about cancellation without expressing a decision", "the speaker describes a possible plan to leave", "the speaker requests cancellation and gives an effective date", "the speaker confirms the decision and asks for the arrangement to be closed")),

    ("compliance", "A", "Control coverage", "coverage of required controls", "how completely the described process includes the controls specified by a policy", (
        "the process omits the required control activities", "the process names a control but provides no operating step", "the process performs the control for a limited part of the workflow", "the process applies the control across ordinary cases with a narrow exception", "the process applies and records each required control across the workflow")),
    ("compliance", "B", "Permission-scope alignment", "alignment with granted permission scope", "how closely the described action stays within the permissions that were granted", (
        "the action exceeds the permissions that were granted", "the action uses a neighboring permission that was not specified", "the action stays within scope only after a separate approval", "the action follows the granted scope with a narrow interpretation", "the action is directly covered by the permission as written")),
    ("compliance", "C", "Retention-policy adherence", "adherence to a retention policy", "how closely stored material follows the stated retention and deletion rules", (
        "records remain after the required deletion event", "some records are removed while related copies remain", "records follow the schedule in routine cases but miss an exception", "records are removed on schedule with a documented narrow delay", "records and their copies follow the retention and deletion rules")),

    ("plausibility", "A", "Causal coherence", "causal coherence of the explanation", "how well the stated causes and effects form a coherent account", (
        "the proposed effect has no causal connection to the stated events", "the account names a possible cause but skips the link to the effect", "the main causal link is possible while an important step remains unexplained", "the account connects the events with a coherent mechanism", "the account's causes, intermediate steps, and effects form a consistent mechanism")),
    ("plausibility", "B", "Claim support", "evidential support for the claim", "how much of the stated claim is backed by the described observations", (
        "the record offers no observation supporting the claim", "the record contains an anecdote that does not address the claim directly", "the record supports a related part of the claim", "the record contains direct observations supporting most of the claim", "the record directly supports the claim across the cases it describes")),
    ("plausibility", "C", "Mechanism feasibility", "feasibility of the proposed mechanism", "how well the proposed process could operate under the stated conditions", (
        "a stated condition prevents the mechanism from operating", "the mechanism could operate only after an unstated resource is added", "the mechanism operates if one uncertain condition is resolved", "the mechanism can operate under the described conditions with a narrow adjustment", "the mechanism can operate as described using the available conditions and resources")),

    ("sentiment", "A", "Positive affect", "intensity of positive affect", "how strongly the language conveys favorable feeling toward its subject", (
        "the speaker expresses no favorable feeling toward the subject", "the speaker offers a restrained favorable remark", "the speaker describes a favorable experience with some emphasis", "the speaker conveys clear enthusiasm about the experience", "the speaker repeatedly expresses strong enthusiasm and delight")),
    ("sentiment", "B", "Frustration affect", "intensity of frustration", "how strongly the language conveys irritation or frustration", (
        "the speaker describes the situation without irritation", "the speaker notes a minor annoyance and continues neutrally", "the speaker describes recurring frustration with the situation", "the speaker uses emphatic language about repeated frustration", "the speaker expresses sustained anger and says the situation is intolerable")),
    ("sentiment", "C", "Expressed confidence", "strength of expressed confidence", "how strongly a speaker communicates confidence in a stated belief", (
        "the speaker openly doubts the stated belief", "the speaker considers the belief possible but uncertain", "the speaker leans toward the belief while noting unresolved doubt", "the speaker states the belief with clear confidence", "the speaker firmly stands by the belief and rejects the stated doubt")),

    ("similarity", "A", "Observable-trait overlap", "overlap in observable traits", "how many described characteristics two items visibly share", (
        "the items share no described visible characteristic", "the items share a superficial feature while their other traits differ", "the items share several traits alongside clear differences", "the items share most of their described traits", "the items match across nearly every described visible trait")),
    ("similarity", "B", "Workflow-sequence resemblance", "resemblance between workflow sequences", "how closely two described workflows follow the same sequence of steps", (
        "the workflows use different steps in different orders", "the workflows share an opening step but then diverge", "the workflows share a central sequence with different handoffs", "the workflows follow the same sequence with a narrow branch", "the workflows follow the same sequence and handoffs throughout")),
    ("similarity", "C", "Outcome-pattern resemblance", "resemblance between outcome patterns", "how closely two described results share their pattern of changes and effects", (
        "the outcomes change in unrelated ways", "the outcomes share one broad effect while their patterns diverge", "the outcomes share a recurring pattern among distinct effects", "the outcomes track each other across most described changes", "the outcomes show the same pattern across the described changes and effects")),

    ("priority", "A", "Deadline precedence", "precedence created by a deadline", "how strongly a deadline moves an item ahead of other work", (
        "the deadline leaves the item in the ordinary queue", "the deadline warrants attention after routine scheduled work", "the deadline places the item ahead of routine work", "the deadline places the item ahead of most pending work", "the deadline requires the item to be handled before other work proceeds")),
    ("priority", "B", "Strategic consequence weight", "weight of strategic consequences", "how strongly an item affects a stated long-range objective", (
        "the item has no stated connection to the long-range objective", "the item supports a peripheral activity around the objective", "the item affects one important component of the objective", "the item influences several important components of the objective", "the item's outcome determines whether the central objective can proceed")),
    ("priority", "C", "Queue elevation", "elevation within a service queue", "how far a request is advanced relative to other work in the queue", (
        "the request remains behind routine queued work", "the request advances only after ordinary queue rules are applied", "the request moves ahead of routine work when a review occurs", "the request is placed near the front for the next available handler", "the request is handled before the ordinary queue is served")),

    ("trust", "A", "Provenance traceability", "traceability of a record's origin", "how fully the origin and handling history of a record can be followed", (
        "the record has no recoverable origin or handling history", "the record identifies a source but not the path it followed", "the record identifies its source and several handling steps", "the record's source and nearly all handling steps can be followed", "the record has a continuous, reviewable history from origin to present")),
    ("trust", "B", "Source historical reliability", "reliability of a source's past reporting", "how consistently a source's past reports have matched later checks", (
        "later checks repeatedly conflict with the source's reports", "later checks confirm isolated reports but contradict others", "later checks confirm the central reports with recurring exceptions", "later checks usually confirm reports across different cases", "later checks consistently confirm the source's reports across cases")),
    ("trust", "C", "Independent corroboration", "breadth of independent corroboration", "how many independent lines of evidence support the same stated account", (
        "the account has no independent corroborating source", "one source repeats the account without checking it independently", "a separate source confirms one part of the account", "several independent sources confirm the central account", "independent records converge on the account and its key details")),

    ("effort", "A", "Manual handling burden", "amount of manual handling required", "how much hands-on work a person must perform to complete the process", (
        "the process completes without manual handling", "a person performs a brief check before completion", "a person handles several routine steps", "a person must repeatedly handle materials across the process", "a person performs hands-on work through nearly every stage")),
    ("effort", "B", "Coordination burden", "amount of coordination required", "how much arranging and follow-up is needed among people or teams", (
        "the task proceeds without coordination between people", "one brief handoff completes the coordination", "several planned handoffs are needed", "teams repeatedly arrange timing and resolve handoff conflicts", "completion depends on continuous coordination across teams")),
    ("effort", "C", "Cognitive step burden", "number and complexity of reasoning steps", "how much deliberate reasoning and decision tracking a person must sustain", (
        "the task needs no tracking beyond a direct action", "the task needs a simple check against one known condition", "the task requires several linked judgments", "the task requires tracking interacting conditions across stages", "the task requires sustained reasoning over interdependent decisions and exceptions")),

    ("latency", "A", "Response waiting interval", "length of the response waiting interval", "how long a requester waits before receiving a response", (
        "the response arrives before the requester waits", "the response arrives after a brief pause", "the requester waits through a routine interval", "the response arrives after a prolonged wait that disrupts follow-up", "the response arrives only after the requester has abandoned the active task")),
    ("latency", "B", "Transfer delay", "delay during data transfer", "how much time transfer adds before the receiving side can use the data", (
        "transferred data is usable as soon as it is sent", "a short transfer pause precedes use", "a transfer delay recurs during ordinary exchanges", "transfer delays interrupt several downstream activities", "transferred data arrives after the receiving work has moved on")),
    ("latency", "C", "Processing queue delay", "delay in the processing queue", "how much time work spends waiting before processing begins", (
        "work begins processing as soon as it enters the system", "work waits behind a brief active task", "work waits through the usual queue cycle", "work remains queued across repeated processing cycles", "work is still waiting after its intended processing window has passed")),

    ("durability", "A", "Abrasion resistance", "resistance to surface abrasion", "how well a surface retains its condition under repeated rubbing or contact", (
        "ordinary contact quickly removes the surface finish", "light contact leaves visible wear", "repeated contact wears the surface while leaving its base intact", "the surface retains its finish through routine contact and shows wear after extended use", "the surface remains intact through repeated contact and extended use")),
    ("durability", "B", "Moisture endurance", "endurance under moisture exposure", "how well a component continues working after exposure to moisture", (
        "moisture exposure stops the component from working", "the component works only while kept away from routine moisture", "the component tolerates brief splashes but not lingering dampness", "the component continues working through ordinary damp conditions", "the component continues working after sustained exposure to the moisture described")),
    ("durability", "C", "Repair persistence", "persistence of a repair under continued use", "how long a completed repair remains effective during ordinary continued use", (
        "the repaired fault returns as soon as ordinary use resumes", "the repair holds through a short return to use before the fault returns", "the repair lasts through routine use but fails under recurring load", "the repair holds under recurring load with occasional adjustment", "the repair remains effective through the continued use described")),

    ("reliability", "A", "Activation continuity", "continuity of successful activation", "how consistently a system starts successfully when it is activated", (
        "activation repeatedly fails before the system starts", "activation succeeds only after another attempt", "activation succeeds in routine cases but fails in familiar exceptions", "activation succeeds across routine and most exceptional cases", "activation succeeds consistently across the described cases")),
    ("reliability", "B", "Interruption recovery", "consistency of recovery after interruption", "how consistently operation resumes after an interruption", (
        "operation does not resume after an interruption", "operation resumes only after manual reconstruction", "operation resumes after routine interruptions but loses some state", "operation resumes with its state intact after the interruptions described", "operation resumes consistently with its state intact across interruptions")),
    ("reliability", "C", "Calibration drift", "extent of calibration drift over continued use", "how far a calibrated output moves away from its reference during use", (
        "the output stays aligned with its reference during continued use", "a small shift appears after extended operation", "the shift recurs and needs occasional recalibration", "the output drifts through ordinary operation and requires repeated recalibration", "the output departs substantially from its reference during routine operation")),

    ("privacy", "A", "Record identifiability", "identifiability of stored records", "how readily a stored record can be connected to a particular person", (
        "the stored record cannot be linked to a person from its retained details", "linkage requires a separate record not held with the data", "the retained details narrow the record to a small group", "the record's retained details identify a person in ordinary review", "the record directly names the person alongside the stored activity")),
    ("privacy", "B", "Personal-data retention duration", "duration of personal-data retention", "how long personal data remains available after its operational use", (
        "personal data is removed when its operational use ends", "personal data remains only through a short closeout period", "personal data remains through the ordinary review cycle", "personal data remains beyond the review cycle for later reuse", "personal data remains available without a defined end to its retention")),
    ("privacy", "C", "External disclosure reach", "reach of external data disclosure", "how widely personal data is shared beyond the organization that collected it", (
        "personal data stays within the collecting team", "personal data reaches one named service provider", "personal data is shared with several contracted providers", "personal data is passed among partner organizations for related uses", "personal data is made available across a broad external network")),

    ("business impact", "A", "Customer departure exposure", "exposure to customer departure", "how strongly a disruption or decision can lead customers to leave", (
        "the described event does not affect customer continuation", "a customer raises a concern but continues the arrangement", "some customers reconsider continuation after the event", "customer departures follow recurring instances of the event", "the event places continued customer relationships at immediate risk")),
    ("business impact", "B", "Revenue concentration dependence", "dependence on a concentrated revenue source", "how much ongoing revenue depends on a narrow set of customers or channels", (
        "revenue comes from a broad mix of independent sources", "one source contributes a visible but replaceable share", "several sources depend on the same underlying account or channel", "a large part of revenue depends on a narrow source group", "the business cannot sustain ordinary operation if its central source leaves")),
    ("business impact", "C", "Operational disruption reach", "reach of operational disruption", "how broadly a disruption affects connected business operations", (
        "the disruption stays within one isolated activity", "the disruption delays one adjacent activity", "the disruption interrupts several connected activities", "the disruption affects most teams that depend on the process", "the disruption prevents connected operations from continuing")),
)

PAGE_TEMPLATES: Mapping[str, tuple[str, str, str]] = {
    "train": (
        "Routine logs note that {clause}.",
        "An operating note records that {clause}.",
        "A case note reports that {clause}.",
    ),
    "validation": (
        "Inspection summaries say that {clause}.",
        "Field reviews record that {clause}.",
        "A follow-up report notes that {clause}.",
    ),
    "calibration": (
        "Evaluation records indicate that {clause}.",
        "During a routine check, staff observed that {clause}.",
        "An audit note states that {clause}.",
    ),
    "development": (
        "Team notes describe how {clause}.",
        "An operator's record says that {clause}.",
        "Review notes state that {clause}.",
    ),
    "final": (
        "Service records indicate that {clause}.",
        "An on-site report describes how {clause}.",
        "A case summary records that {clause}.",
    ),
}

QUESTION_TEMPLATES: Mapping[str, Mapping[int, tuple[str, str, str]]] = {
    "train": {
        1: ("Which option shows more {criterion}?", "Where is {criterion} expressed more strongly?", "Which alternative shows a greater degree of {criterion}?"),
        -1: ("Which option shows less {criterion}?", "Where is {criterion} expressed less strongly?", "Which alternative shows a lesser degree of {criterion}?"),
    },
    "validation": {
        1: ("Between these options, where does the evidence show more {criterion}?", "Which record displays a greater extent of {criterion}?", "In which alternative is {criterion} more apparent?"),
        -1: ("Between these options, where does the evidence show less {criterion}?", "Which record displays a lesser extent of {criterion}?", "In which alternative is {criterion} less apparent?"),
    },
    "calibration": {
        1: ("Which account provides stronger evidence of {criterion}?", "Where is {criterion} more pronounced across the alternatives?", "Which option exhibits a greater degree of {criterion}?"),
        -1: ("Which account provides weaker evidence of {criterion}?", "Where is {criterion} less pronounced across the alternatives?", "Which option exhibits a lesser degree of {criterion}?"),
    },
    "development": {
        1: ("Reading the evidence, which option shows a greater extent of {criterion}?", "Which record makes {criterion} more evident?", "Where is {criterion} more strongly represented?"),
        -1: ("Reading the evidence, which option shows a lesser extent of {criterion}?", "Which record makes {criterion} less evident?", "Where is {criterion} less strongly represented?"),
    },
    "final": {
        1: ("On this question, which record shows more {criterion}?", "Where is a greater extent of {criterion} evident?", "Which option provides stronger evidence of {criterion}?"),
        -1: ("On this question, which record shows less {criterion}?", "Where is a lesser extent of {criterion} evident?", "Which option provides weaker evidence of {criterion}?"),
    },
}


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:length]}"


def _protocol() -> tuple[dict[str, Any], bytes, str]:
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("experiment") != "ECA-1 criterion-conditioned Evidence Pages":
        raise ValueError("unexpected ECA protocol identity")
    root = Path(protocol["output_root"]).expanduser().resolve()
    expected = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1")
    if root != expected:
        raise ValueError(f"protocol output_root changed: expected {expected}, found {root}")
    if tuple(protocol["corpus"]["families"]) != FAMILIES:
        raise ValueError("protocol family order differs from the authored schedule")
    if protocol["corpus"]["worlds"] != WORLD_COUNTS:
        raise ValueError("protocol world counts differ from the builder schedule")
    if protocol["canonical_dependency"]["root"] != str(CANONICAL_ROOT):
        raise ValueError("canonical dependency root changed")
    for relative, expected_hash in EXPECTED_CANONICAL_FILES.items():
        pinned = protocol["canonical_dependency"]["files_sha256"].get(relative)
        if pinned != expected_hash:
            raise ValueError(f"protocol canonical hash pin changed for {relative}")
        actual, _ = _sha_file(CANONICAL_ROOT / relative)
        if actual != expected_hash:
            raise ValueError(f"canonical dependency hash mismatch: {relative}")
    return protocol, raw, root


def _private_directory(path: Path, *, create: bool) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"private path is not a real directory: {path}")
        if path.stat().st_uid != os.getuid():
            raise ValueError(f"private directory is not owned by current user: {path}")
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError(f"private directory permissions must exclude group/other access: {path}")
    elif create:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    else:
        raise FileNotFoundError(path)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Preserve the exclusive partial file as a failure artifact; never
        # overwrite or silently remove source material.
        raise
    return _sha_bytes(payload)


def _normal_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _make_source() -> tuple[dict[str, Any], dict[str, Any]]:
    if len(AUTHORED_FIELDS) != 60:
        raise ValueError(f"expected 60 authored properties, found {len(AUTHORED_FIELDS)}")
    per_family = Counter(row[0] for row in AUTHORED_FIELDS)
    if tuple(per_family.keys()) != FAMILIES or any(per_family[name] != 3 for name in FAMILIES):
        raise ValueError("the source catalogue must contain exactly three fields per protocol family")
    properties: list[dict[str, Any]] = []
    for family, field_code, title, criterion, description, clauses in AUTHORED_FIELDS:
        property_id = _stable_id("f", family, field_code, title, length=20)
        if len(clauses) != 5:
            raise ValueError(f"property {title} does not have five grade descriptions")
        wordings: list[dict[str, Any]] = []
        for grade, clause in enumerate(clauses):
            for split in SPLITS:
                for variant, template in enumerate(PAGE_TEMPLATES[split]):
                    text = template.format(clause=clause)
                    if re.search(r"\b(?:grade|level)\s*\d|\b[0-4]\b", text, re.IGNORECASE):
                        raise ValueError(f"numeric or literal grade token in authored page text: {title}")
                    wordings.append({
                        "wording_id": _stable_id("w", property_id, split, grade, variant),
                        "split": split, "grade": grade, "variant": variant, "text": text,
                    })
        questions: list[dict[str, Any]] = []
        for split in SPLITS:
            for orientation in (1, -1):
                for variant, template in enumerate(QUESTION_TEMPLATES[split][orientation]):
                    text = template.format(criterion=criterion)
                    questions.append({
                        "question_id": _stable_id("q", property_id, split, orientation, variant),
                        "split": split, "orientation": orientation, "variant": variant, "text": text,
                    })
        properties.append({
            "property_id": property_id, "family": family, "field_code": field_code,
            "title": title, "criterion": criterion, "description": description,
            "grade_order": list(GRADE_ORDER), "grade_rubrics": list(clauses),
            "page_wordings": wordings, "questions": questions,
        })

    source = {
        "schema": SOURCE_SCHEMA,
        "seed": SEED,
        "grade_order": list(GRADE_ORDER),
        "families": list(FAMILIES),
        "split_field_codes": {"train": "A", "validation": "A", "calibration": "A", "development": "B", "final": "C"},
        "properties": properties,
        "authorship_contract": {
            "extent_not_desirability": True,
            "evidence_has_no_numeric_or_literal_grade_tokens": True,
            "page_variants_per_grade_per_split": 3,
            "question_variants_per_orientation_per_split": 3,
            "opaque_metadata_not_semantic_text": True,
        },
    }
    source_bytes = _canonical_json(source)
    source_sha = _sha_bytes(source_bytes)
    page_text_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    question_text_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    page_template_hashes = {split: {_sha_bytes(item.encode()) for item in PAGE_TEMPLATES[split]} for split in SPLITS}
    question_template_hashes = {
        split: {_sha_bytes(item.encode()) for orient in (1, -1) for item in QUESTION_TEMPLATES[split][orient]}
        for split in SPLITS
    }
    for prop in properties:
        for item in prop["page_wordings"]:
            page_text_hashes[item["split"]].add(_sha_bytes(_normal_text(item["text"]).encode("utf-8")))
        for item in prop["questions"]:
            question_text_hashes[item["split"]].add(_sha_bytes(_normal_text(item["text"]).encode("utf-8")))
    _assert_disjoint_hash_sets(page_text_hashes, "page text")
    _assert_disjoint_hash_sets(question_text_hashes, "criterion question")
    _assert_disjoint_hash_sets(page_template_hashes, "page template")
    _assert_disjoint_hash_sets(question_template_hashes, "question template")
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "source_sha256": source_sha,
        "grade_order": list(GRADE_ORDER),
        "families": {family: 3 for family in FAMILIES},
        "property_count": len(properties),
        "grade_rubric_count": sum(len(prop["grade_rubrics"]) for prop in properties),
        "page_wording_count": sum(len(prop["page_wordings"]) for prop in properties),
        "question_count": sum(len(prop["questions"]) for prop in properties),
        "page_wordings_per_property_grade_split": 3,
        "questions_per_property_split_orientation": 3,
        "counts_by_split": {
            split: {
                "page_wordings": sum(len([x for x in p["page_wordings"] if x["split"] == split]) for p in properties),
                "questions": sum(len([x for x in p["questions"] if x["split"] == split]) for p in properties),
                "unique_page_text_sha256": sorted(page_text_hashes[split]),
                "unique_question_sha256": sorted(question_text_hashes[split]),
                "page_template_sha256": sorted(page_template_hashes[split]),
                "question_template_sha256": sorted(question_template_hashes[split]),
            }
            for split in SPLITS
        },
        "cross_split_exact_normalized_text_overlap": 0,
        "cross_split_page_template_overlap": 0,
        "cross_split_question_template_overlap": 0,
    }
    return source, inventory


def _assert_disjoint_hash_sets(by_split: Mapping[str, set[str]], description: str) -> None:
    prior: set[str] = set()
    for split in SPLITS:
        overlap = prior.intersection(by_split[split])
        if overlap:
            raise ValueError(f"cross-split {description} overlap detected: {split}")
        prior.update(by_split[split])


def _make_review_files(source: Mapping[str, Any], source_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    packet_items: list[dict[str, Any]] = []
    key_items: dict[str, dict[str, Any]] = {}
    for prop in source["properties"]:
        for wording in prop["page_wordings"]:
            review_id = _stable_id("r", source_sha, "page", wording["wording_id"], length=28)
            packet_items.append({
                "review_id": review_id,
                "item_type": "page",
                "page_text": wording["text"],
                "neutral_property_description": prop["description"],
                "ordered_extent_rubric": list(prop["grade_rubrics"]),
            })
            key_items[review_id] = {
                "item_type": "page", "property_id": prop["property_id"], "family": prop["family"],
                "field_code": prop["field_code"], "split": wording["split"],
                "grade": wording["grade"], "wording_id": wording["wording_id"],
            }
        for question in prop["questions"]:
            review_id = _stable_id("r", source_sha, "question", question["question_id"], length=28)
            packet_items.append({
                "review_id": review_id,
                "item_type": "question",
                "question": question["text"],
                "neutral_property_description": prop["description"],
            })
            key_items[review_id] = {
                "item_type": "question", "property_id": prop["property_id"], "family": prop["family"],
                "field_code": prop["field_code"], "split": question["split"],
                "orientation": question["orientation"], "question_id": question["question_id"],
            }
    packet_items.sort(key=lambda item: item["review_id"])
    packet = {
        "schema": PACKET_SCHEMA,
        "source_sha256": source_sha,
        "items": packet_items,
        "review_instructions": "For each opaque item, assess whether the page's stated extent fits the ordered neutral rubric, or whether the question clearly asks about its described property. Write one raw JSONL review object per review_id with reviewer_id, review_id, decision (accept, ambiguous, or reject), and optional note. Do not infer unstated facts.",
        "decision_values": ["accept", "ambiguous", "reject"],
    }
    packet_bytes = _canonical_json(packet)
    packet_sha = _sha_bytes(packet_bytes)
    key = {
        "schema": KEY_SCHEMA,
        "source_sha256": source_sha,
        "packet_sha256": packet_sha,
        "grade_order": list(GRADE_ORDER),
        "items": key_items,
    }
    return packet, key


def _prepare_source(root: Path, protocol: Mapping[str, Any], protocol_raw: bytes) -> dict[str, Any]:
    _private_directory(root, create=True)
    source_dir = root / "source"
    audit_dir = root / "audit"
    _private_directory(source_dir, create=True)
    _private_directory(audit_dir, create=True)
    source, inventory = _make_source()
    source_bytes = _canonical_json(source)
    source_sha = _sha_bytes(source_bytes)
    packet, key = _make_review_files(source, source_sha)
    packet_bytes = _canonical_json(packet)
    key_bytes = _canonical_json(key)
    inventory_bytes = _canonical_json(inventory)
    source_path = source_dir / "eca_authored_source_v1.json"
    inventory_path = source_dir / "eca_source_inventory_v1.json"
    packet_path = audit_dir / "opaque_review_packet_v1.json"
    key_path = audit_dir / "sealed_target_key_v1.json"
    source_file_sha = _write_exclusive(source_path, source_bytes)
    inventory_file_sha = _write_exclusive(inventory_path, inventory_bytes)
    packet_file_sha = _write_exclusive(packet_path, packet_bytes)
    key_file_sha = _write_exclusive(key_path, key_bytes)
    protocol_sha = _sha_bytes(protocol_raw)
    code_sha, code_size = _sha_file(Path(__file__).resolve())
    dependency_hashes = {relative: _sha_file(CANONICAL_ROOT / relative)[0] for relative in EXPECTED_CANONICAL_FILES}
    manifest = {
        "schema": "vey.eca1.source-preparation-manifest.v1",
        "source_sha256": source_sha,
        "protocol_sha256": protocol_sha,
        "builder_sha256": code_sha,
        "builder_bytes": code_size,
        "canonical_dependency_sha256": dependency_hashes,
        "files": {
            "source/eca_authored_source_v1.json": {"sha256": source_file_sha, "bytes": len(source_bytes)},
            "source/eca_source_inventory_v1.json": {"sha256": inventory_file_sha, "bytes": len(inventory_bytes)},
            "audit/opaque_review_packet_v1.json": {"sha256": packet_file_sha, "bytes": len(packet_bytes)},
            "audit/sealed_target_key_v1.json": {"sha256": key_file_sha, "bytes": len(key_bytes)},
        },
        "audit_item_count": len(packet["items"]),
        "audit_items_by_type": dict(Counter(item["item_type"] for item in packet["items"])),
        "source_coverage": {key: inventory[key] for key in ("property_count", "grade_rubric_count", "page_wording_count", "question_count", "grade_order", "cross_split_exact_normalized_text_overlap", "cross_split_page_template_overlap", "cross_split_question_template_overlap")},
    }
    manifest_bytes = _canonical_json(manifest)
    _write_exclusive(source_dir / "prepare_manifest_v1.json", manifest_bytes)
    return manifest


@dataclass(frozen=True)
class Term:
    property_id: str
    question: str
    orientation: int
    weight: float


@dataclass(frozen=True)
class QuerySpec:
    query_key: str
    query_kind: str
    terms: tuple[Term, ...]
    question: str


@dataclass(frozen=True)
class World:
    world_id: str
    split: str
    local_index: int
    rank: int
    families: tuple[str, ...]
    properties: tuple[Mapping[str, Any], ...]
    exposure_by_family: Mapping[str, int]


def _world_plan(source: Mapping[str, Any]) -> list[World]:
    property_by_family_code = {(p["family"], p["field_code"]): p for p in source["properties"]}
    available_families = {
        "train": FAMILIES[:12], "validation": FAMILIES[:12], "calibration": FAMILIES[:12],
        "development": FAMILIES[:16], "final": FAMILIES,
    }
    # Candidate world IDs are opaque; the stable rank hash determines split
    # membership, then a balanced cyclic family schedule determines content.
    raw_worlds = [
        _stable_id("world", SEED, slot, length=28)
        for slot in range(sum(WORLD_COUNTS.values()))
    ]
    ranked = sorted(raw_worlds, key=lambda wid: hashlib.sha256(f"eca-world-rank:{SEED}:{wid}".encode()).hexdigest())
    plan: list[World] = []
    exposure_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    cursor = 0
    global_rank = 0
    for split in SPLITS:
        families = available_families[split]
        field_code = source["split_field_codes"][split]
        for local_index in range(WORLD_COUNTS[split]):
            world_id = ranked[cursor]
            cursor += 1
            start = (local_index * 4) % len(families)
            world_families = tuple(families[(start + offset) % len(families)] for offset in range(4))
            if len(set(world_families)) != 4:
                raise ValueError(f"family schedule repeated a family in world {world_id}")
            exposure_by_family = {family: exposure_counts[split][family] for family in world_families}
            exposure_counts[split].update(world_families)
            selected = tuple(property_by_family_code[(family, field_code)] for family in world_families)
            plan.append(World(world_id, split, local_index, global_rank, world_families, selected, exposure_by_family))
            global_rank += 1
    if cursor != len(ranked):
        raise ValueError("world allocation did not consume every opaque ID")
    return plan


def _property_maps(source: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    by_id = {prop["property_id"]: prop for prop in source["properties"]}
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for prop in source["properties"]:
        by_family[prop["family"]].append(prop)
    return by_id, by_family


def _page_wording(prop: Mapping[str, Any], split: str, grade: int, variant: int) -> str:
    matches = [item for item in prop["page_wordings"] if item["split"] == split and item["grade"] == grade and item["variant"] == variant]
    if len(matches) != 1:
        raise ValueError(f"expected one page wording for property {prop['property_id']}, {split}, {grade}, {variant}")
    return matches[0]["text"]


def _criterion_question(prop: Mapping[str, Any], split: str, orientation: int, variant: int) -> str:
    matches = [item for item in prop["questions"] if item["split"] == split and item["orientation"] == orientation and item["variant"] == variant]
    if len(matches) != 1:
        raise ValueError(f"expected one question for property {prop['property_id']}, {split}, {orientation}, {variant}")
    return matches[0]["text"]


def _pick_variant(*parts: object) -> int:
    return int(hashlib.sha256("\0".join(str(x) for x in parts).encode()).hexdigest()[:8], 16) % 3


def _make_query(world: World, prop_by_id: Mapping[str, Mapping[str, Any]], query_key: str, query_kind: str, property_orientations: Sequence[tuple[str, int]], variant_salt: str) -> QuerySpec:
    terms: list[Term] = []
    for prop_id, orientation in property_orientations:
        prop = prop_by_id[prop_id]
        variant = _pick_variant(world.world_id, query_key, prop_id, orientation, variant_salt)
        terms.append(Term(
            property_id=prop_id,
            question=_criterion_question(prop, world.split, orientation, variant),
            orientation=orientation,
            weight=1.0 if len(property_orientations) == 1 else 0.5,
        ))
    question = terms[0].question if len(terms) == 1 else "; also ".join(term.question for term in terms)
    return QuerySpec(query_key, query_kind, tuple(terms), question)


def _query_specs(world: World, prop_by_id: Mapping[str, Mapping[str, Any]], by_family: Mapping[str, list[Mapping[str, Any]]]) -> tuple[list[QuerySpec], Mapping[str, str]]:
    present = list(world.properties)
    specs: list[QuerySpec] = []
    for prop in present:
        for orientation in (1, -1):
            key = f"atomic:{prop['property_id']}:{orientation:+d}"
            specs.append(_make_query(world, prop_by_id, key, "atomic", ((prop["property_id"], orientation),), "main"))
    eligible = FAMILIES[:12] if world.split in ("train", "validation", "calibration") else FAMILIES[:16] if world.split == "development" else FAMILIES
    absent_family = next((family for family in eligible if family not in world.families), None)
    if absent_family is None:
        raise ValueError(f"no absent family available in world {world.world_id}")
    field_code = "A" if world.split in ("train", "validation", "calibration") else "B" if world.split == "development" else "C"
    absent_prop = next(prop for prop in by_family[absent_family] if prop["field_code"] == field_code)
    absent_key = f"absent:{absent_prop['property_id']}"
    specs.append(_make_query(world, prop_by_id, absent_key, "absent", ((absent_prop["property_id"], 1),), "absent"))
    first, second = present[:2]
    comp_key = f"composition:{first['property_id']}:{second['property_id']}:positive"
    specs.append(_make_query(world, prop_by_id, comp_key, "composition", ((first["property_id"], 1), (second["property_id"], 1)), "composition-positive"))
    reverse_key = f"composition:{second['property_id']}:{first['property_id']}:full-reversal"
    specs.append(_make_query(world, prop_by_id, reverse_key, "composition_reversal", ((second["property_id"], -1), (first["property_id"], -1)), "composition-reversal"))
    return specs, {"absent_family": absent_family, "absent_property_id": absent_prop["property_id"]}


def _exact_values(world_id: str, candidate_id: str) -> dict[str, int]:
    digest = hashlib.sha256(f"eca-exact:{SEED}:{world_id}:{candidate_id}".encode()).digest()
    return {
        "setup_charge_cents": 275 + int.from_bytes(digest[0:2], "big") % 8725,
        "handoff_minutes": 3 + int.from_bytes(digest[2:4], "big") % 478,
        "replacement_parts": int.from_bytes(digest[4:6], "big") % 13,
    }


def _exact_block_text(facts: Mapping[str, int]) -> str:
    return (
        f"Quoted setup charge {facts['setup_charge_cents']} cents; "
        f"handoff delay {facts['handoff_minutes']} minutes; "
        f"replacement parts {facts['replacement_parts']} units."
    )


def _base_candidate_id(world_id: str, ordinal: int) -> str:
    return _stable_id("cand", world_id, ordinal, length=12)


def _build_world_state(world: World, split_properties: Sequence[Mapping[str, Any]], candidate_count: int = 4) -> tuple[tuple[Candidate, ...], tuple[StateBlock, ...], dict[str, str], dict[str, int], dict[str, str], dict[str, dict[str, int]], dict[str, int]]:
    candidates: list[Candidate] = []
    blocks: list[StateBlock] = []
    owners: dict[str, str] = {}
    grades: dict[str, int] = {}
    fields: dict[str, str] = {}
    exact_by_candidate: dict[str, dict[str, int]] = {}
    stable_ordinals: dict[str, int] = {}
    for candidate_index in range(candidate_count):
        cid = _base_candidate_id(world.world_id, candidate_index) if candidate_index < 4 else _stable_id("cand", world.world_id, "probe", candidate_index, length=12)
        candidates.append(Candidate(cid, "Alternative record."))
        stable_ordinals[cid] = candidate_index
        exact = _exact_values(world.world_id, cid)
        exact_by_candidate[cid] = exact
        for prop in split_properties:
            exposure = world.exposure_by_family[prop["family"]]
            grade = (4 * exposure + candidate_index) % 5
            variant = (world.local_index + candidate_index + FAMILIES.index(prop["family"])) % 3
            text = _page_wording(prop, world.split, grade, variant)
            block_id = _stable_id("page", world.world_id, cid, prop["property_id"], "primary", length=18)
            blocks.append(StateBlock(block_id, text, kind="semantic_page", exact=False))
            owners[block_id] = cid
            grades[block_id] = grade
            fields[block_id] = prop["property_id"]
        exact_id = _stable_id("exact", world.world_id, cid, length=18)
        blocks.append(StateBlock(exact_id, _exact_block_text(exact), kind="typed_exact_facts", exact=True))
    return tuple(candidates), tuple(blocks), owners, grades, fields, exact_by_candidate, stable_ordinals


def _term_metadata(terms: Sequence[Term]) -> list[dict[str, Any]]:
    return [{"question": term.question, "field_key": term.property_id, "orientation": term.orientation, "weight": term.weight} for term in terms]


def _score_candidates(candidates: Sequence[Candidate], owners: Mapping[str, str], grades: Mapping[str, int], fields: Mapping[str, str], terms: Sequence[Term]) -> tuple[dict[str, bool], dict[str, float | None], tuple[str, ...], float | None, str | None]:
    known: dict[str, bool] = {}
    scores: dict[str, float | None] = {}
    required_unknown = False
    for candidate in candidates:
        if candidate.id == UNKNOWN_ID:
            known[candidate.id] = False
            scores[candidate.id] = None
            continue
        utilities: list[float] = []
        valid = True
        for term in terms:
            page_ids = [block_id for block_id, owner in owners.items() if owner == candidate.id and fields.get(block_id) == term.property_id]
            if len(page_ids) != 1 or page_ids[0] not in grades:
                valid = False
                break
            grade = grades[page_ids[0]]
            utilities.append((grade / 4.0) if term.orientation == 1 else ((4 - grade) / 4.0))
        known[candidate.id] = valid
        if not valid:
            required_unknown = True
            scores[candidate.id] = None
        else:
            scores[candidate.id] = sum(term.weight * utility for term, utility in zip(terms, utilities))
    evidence_ids = [candidate.id for candidate in candidates if candidate.id != UNKNOWN_ID]
    if required_unknown or not evidence_ids:
        return known, scores, (UNKNOWN_ID,), None, "required_evidence_missing_or_conflicting"
    values = [scores[cid] for cid in evidence_ids]
    if any(value is None for value in values):
        return known, scores, (UNKNOWN_ID,), None, "required_evidence_missing_or_conflicting"
    best = max(float(value) for value in values if value is not None)
    winners = tuple(cid for cid in evidence_ids if math.isclose(float(scores[cid]), best, rel_tol=0.0, abs_tol=1e-12))
    return known, scores, winners, best, None


def _make_row(world: World, query: QuerySpec, candidates: Sequence[Candidate], blocks: Sequence[StateBlock], owners: Mapping[str, str], grades: Mapping[str, int], fields: Mapping[str, str], exact_by_candidate: Mapping[str, Mapping[str, int]], stable_ordinals: Mapping[str, int], *, variant: str, parent_id: str | None = None, extra_metadata: Mapping[str, Any] | None = None) -> DecisionIR:
    output_candidates = tuple(candidates) + (Candidate(UNKNOWN_ID, "Insufficient evidence"),)
    known, teacher_scores, gold, gold_scalar, unknown_reason = _score_candidates(output_candidates, owners, grades, fields, query.terms)
    evidence = tuple(Evidence(block_id) for block_id, property_id in fields.items() if property_id in {term.property_id for term in query.terms})
    family_for_term = {term.property_id: next(prop["family"] for prop in world.properties if prop["property_id"] == term.property_id) for term in query.terms if any(prop["property_id"] == term.property_id for prop in world.properties)}
    metadata: dict[str, Any] = {
        "page_owners": dict(owners),
        "page_grades": dict(grades),
        "page_fields": dict(fields),
        "terms": _term_metadata(query.terms),
        "known": known,
        "teacher_scores": teacher_scores,
        "world_id": world.world_id,
        "source_world_id": world.world_id,
        "variant": variant,
        "family": sorted(set(family_for_term.values()) or set(world.families)),
        "gold_scalar": gold_scalar,
        "unknown_reason": unknown_reason,
        "query_kind": query.query_kind,
        "query_id": query.query_key,
        "stable_ordinals": dict(stable_ordinals),
        "exact_facts": {candidate_id: dict(values) for candidate_id, values in exact_by_candidate.items()},
        "exact_fact_keys": ["setup_charge_cents", "handoff_minutes", "replacement_parts"],
        "candidate_count": len(candidates),
        "independent_observation": variant == "base",
    }
    if parent_id is not None:
        metadata["parent_decision_id"] = parent_id
    if extra_metadata:
        metadata.update(extra_metadata)
    row_id = _stable_id("eca", world.world_id, query.query_key, variant, query.question, *(candidate.id for candidate in candidates))
    return DecisionIR(
        id=row_id,
        source="eca1_authored",
        split=world.split,
        task="choice",
        state_blocks=tuple(blocks),
        question=query.question,
        candidates=output_candidates,
        gold=gold,
        license="New authored shipping-train material only; pretrained checkpoint remains conditional/review",
        evidence=evidence,
        constraints=(),
        metadata=metadata,
    )


def _world_family_for_property(world: World, property_id: str) -> str | None:
    for prop in world.properties:
        if prop["property_id"] == property_id:
            return prop["family"]
    return None


def _spec_family_map(world: World, prop_by_id: Mapping[str, Mapping[str, Any]], terms: Sequence[Term]) -> dict[str, str]:
    result = {}
    for term in terms:
        prop = prop_by_id[term.property_id]
        result[term.property_id] = prop["family"]
    return result


def _replace_row(row: DecisionIR, *, candidates: Sequence[Candidate] | None = None, blocks: Sequence[StateBlock] | None = None, question: str | None = None, gold: Sequence[str] | None = None, metadata: Mapping[str, Any] | None = None, evidence: Sequence[Evidence] | None = None, row_variant: str | None = None) -> DecisionIR:
    new_metadata = dict(row.metadata if metadata is None else metadata)
    variant = row_variant or str(new_metadata.get("variant", "variant"))
    new_metadata["variant"] = variant
    candidate_seq = tuple(row.candidates if candidates is None else candidates)
    question_text = row.question if question is None else question
    new_gold = tuple(row.gold if gold is None else gold)
    new_blocks = tuple(row.state_blocks if blocks is None else blocks)
    id_distinction = new_metadata.get("id_distinction", "")
    rid = _stable_id("eca", new_metadata.get("world_id"), new_metadata.get("query_id"), variant, id_distinction, question_text, *(c.id for c in candidate_seq if c.id != UNKNOWN_ID))
    return DecisionIR(
        id=rid, source=row.source, split=row.split, task=row.task,
        state_blocks=new_blocks, question=question_text, candidates=candidate_seq,
        gold=new_gold, license=row.license, evidence=tuple(row.evidence if evidence is None else evidence),
        constraints=row.constraints, metadata=new_metadata,
    )


def _recalculate(row: DecisionIR, *, variant: str, question: str | None = None, terms: Sequence[Term] | None = None, extra_metadata: Mapping[str, Any] | None = None) -> DecisionIR:
    source_terms = row.metadata["terms"] if terms is None else _term_metadata(terms)
    term_objs = tuple(
        Term(item["field_key"], item["question"], int(item["orientation"]), float(item["weight"]))
        for item in source_terms
    )
    owners = row.metadata["page_owners"]
    grades = row.metadata["page_grades"]
    fields = row.metadata["page_fields"]
    known, scores, gold, gold_scalar, reason = _score_candidates(row.candidates, owners, grades, fields, term_objs)
    candidate_ids = {candidate.id for candidate in row.candidates}
    evidence_ids = tuple(Evidence(block_id) for block_id, field_key in fields.items() if field_key in {term.property_id for term in term_objs})
    metadata = dict(row.metadata)
    metadata.update({
        "terms": list(source_terms), "known": known, "teacher_scores": scores,
        "gold_scalar": gold_scalar, "unknown_reason": reason, "variant": variant,
        "independent_observation": False,
    })
    if extra_metadata:
        metadata.update(extra_metadata)
    return _replace_row(row, question=question, gold=gold, metadata=metadata, evidence=evidence_ids, row_variant=variant)


def _grade_change_variant(row: DecisionIR, term_index: int, prop_by_id: Mapping[str, Mapping[str, Any]], split: str) -> DecisionIR:
    terms = row.metadata["terms"]
    term = terms[term_index]
    field_key = term["field_key"]
    orientation = int(term["orientation"])
    owners = dict(row.metadata["page_owners"])
    grades = dict(row.metadata["page_grades"])
    fields = dict(row.metadata["page_fields"])
    page_by_candidate = {
        candidate.id: [block_id for block_id, owner in owners.items() if owner == candidate.id and fields[block_id] == field_key]
        for candidate in row.candidates if candidate.id != UNKNOWN_ID
    }
    possible: list[tuple[float, str, str, int, int]] = []
    for candidate_id, page_ids in page_by_candidate.items():
        if len(page_ids) != 1:
            continue
        block_id = page_ids[0]
        old_grade = grades[block_id]
        new_grade = old_grade + (1 if orientation == 1 else -1)
        if new_grade not in GRADE_ORDER:
            continue
        utility = old_grade / 4.0 if orientation == 1 else (4 - old_grade) / 4.0
        possible.append((utility, candidate_id, block_id, old_grade, new_grade))
    if not possible:
        raise ValueError(f"no relevant-page grade change possible for decision {row.id}")
    _, candidate_id, block_id, old_grade, new_grade = min(possible, key=lambda item: (item[0], hashlib.sha256(item[1].encode()).hexdigest()))
    prop = prop_by_id[field_key]
    variant_idx = _pick_variant(row.id, field_key, candidate_id, new_grade, "intervention")
    replacement_text = _page_wording(prop, split, new_grade, variant_idx)
    blocks = tuple(replace(block, text=replacement_text) if block.id == block_id else block for block in row.state_blocks)
    grades[block_id] = new_grade
    metadata = dict(row.metadata)
    metadata.update({"page_owners": owners, "page_grades": grades, "page_fields": fields})
    intermediate = _replace_row(row, blocks=blocks, metadata=metadata, row_variant="grade_change_source")
    return _recalculate(intermediate, variant="relevant_page_grade_change", extra_metadata={
        "intervention_type": "relevant_page_grade_change",
        "intervened_field_key": field_key,
        "intervened_candidate_id": candidate_id,
        "intervened_page_block_id": block_id,
        "old_grade": old_grade,
        "new_grade": new_grade,
        "id_distinction": f"grade:{field_key}",
    })


def _erase_variant(row: DecisionIR, field_key: str) -> DecisionIR:
    owners = dict(row.metadata["page_owners"])
    grades = dict(row.metadata["page_grades"])
    fields = dict(row.metadata["page_fields"])
    erased = {block_id for block_id, key in fields.items() if key == field_key}
    blocks = tuple(block for block in row.state_blocks if block.id not in erased)
    for block_id in erased:
        owners.pop(block_id, None)
        grades.pop(block_id, None)
        fields.pop(block_id, None)
    metadata = dict(row.metadata)
    metadata.update({"page_owners": owners, "page_grades": grades, "page_fields": fields})
    intermediate = _replace_row(row, blocks=blocks, metadata=metadata, row_variant="erase_source",
                               evidence=tuple(item for item in row.evidence if item.block_id not in erased))
    return _recalculate(intermediate, variant="relevant_page_erasure", extra_metadata={
        "intervention_type": "relevant_page_erasure", "intervened_field_key": field_key,
        "erased_page_count": len(erased), "id_distinction": f"erase:{field_key}",
    })


def _contradiction_variant(row: DecisionIR, field_key: str, split: str, prop_by_id: Mapping[str, Mapping[str, Any]]) -> DecisionIR:
    owners = dict(row.metadata["page_owners"])
    grades = dict(row.metadata["page_grades"])
    fields = dict(row.metadata["page_fields"])
    candidates = [candidate.id for candidate in row.candidates if candidate.id != UNKNOWN_ID]
    target_id = min(candidates, key=lambda cid: hashlib.sha256(f"{row.id}:contradiction:{field_key}:{cid}".encode()).hexdigest())
    matching = [block_id for block_id, owner in owners.items() if owner == target_id and fields[block_id] == field_key]
    if len(matching) != 1:
        raise ValueError(f"contradiction target has no unique relevant page: {row.id}")
    original_block = matching[0]
    old_grade = grades[original_block]
    new_grade = (old_grade + 1) % 5
    prop = prop_by_id[field_key]
    variant_idx = _pick_variant(row.id, field_key, target_id, new_grade, "contradiction")
    text = _page_wording(prop, split, new_grade, variant_idx)
    block_id = _stable_id("page", row.id, field_key, target_id, "contradiction", length=18)
    blocks = tuple(row.state_blocks) + (StateBlock(block_id, text, kind="semantic_page", exact=False),)
    owners[block_id] = target_id
    grades[block_id] = new_grade
    fields[block_id] = field_key
    metadata = dict(row.metadata)
    metadata.update({"page_owners": owners, "page_grades": grades, "page_fields": fields})
    intermediate = _replace_row(row, blocks=blocks, metadata=metadata, row_variant="contradiction_source")
    return _recalculate(intermediate, variant="relevant_page_contradiction", extra_metadata={
        "intervention_type": "relevant_page_contradiction",
        "intervened_field_key": field_key,
        "intervened_candidate_id": target_id,
        "original_page_block_id": original_block,
        "contradictory_page_block_id": block_id,
        "original_grade": old_grade,
        "contradictory_grade": new_grade,
        "id_distinction": f"contra:{field_key}",
    })


def _candidate_permutation(row: DecisionIR, variant_index: int) -> DecisionIR:
    evidence_candidates = [candidate for candidate in row.candidates if candidate.id != UNKNOWN_ID]
    if len(evidence_candidates) > 1:
        shift = variant_index % len(evidence_candidates)
        ranking = evidence_candidates[shift:] + evidence_candidates[:shift]
    else:
        ranking = list(evidence_candidates)
    ranking.append(next(candidate for candidate in row.candidates if candidate.id == UNKNOWN_ID))
    metadata = dict(row.metadata)
    metadata["candidate_order"] = [candidate.id for candidate in ranking]
    metadata["permutation_index"] = variant_index
    metadata["identity_map"] = {candidate.id: candidate.id for candidate in evidence_candidates}
    return _replace_row(row, candidates=ranking, metadata=metadata, row_variant=f"candidate_permutation_{variant_index}")


def _rename_candidates(row: DecisionIR) -> DecisionIR:
    remap = {
        candidate.id: _stable_id("renamed", row.id, candidate.id, length=12)
        for candidate in row.candidates if candidate.id != UNKNOWN_ID
    }
    candidates = tuple(Candidate(remap.get(candidate.id, candidate.id), candidate.description) for candidate in row.candidates)
    metadata = dict(row.metadata)
    metadata["page_owners"] = {
        block_id: remap[owner] for block_id, owner in metadata["page_owners"].items()
    }
    for key in ("known", "teacher_scores", "stable_ordinals", "exact_facts"):
        metadata[key] = {remap.get(candidate_id, candidate_id): value for candidate_id, value in metadata[key].items()}
    metadata["identity_map"] = {new_id: old_id for old_id, new_id in remap.items()}
    metadata["candidate_order"] = [candidate.id for candidate in candidates]
    gold = tuple(remap.get(candidate_id, candidate_id) for candidate_id in row.gold)
    return _replace_row(row, candidates=candidates, gold=gold, metadata=metadata, row_variant="candidate_rename")


def _page_reorder(row: DecisionIR) -> DecisionIR:
    semantic = [block for block in row.state_blocks if not block.exact]
    exact = [block for block in row.state_blocks if block.exact]
    semantic.sort(key=lambda block: hashlib.sha256(f"{row.id}:page-order:{block.id}".encode()).hexdigest())
    if len(semantic) > 1 and [block.id for block in semantic] == [block.id for block in row.state_blocks if not block.exact]:
        semantic = semantic[1:] + semantic[:1]
    return _replace_row(row, blocks=tuple(semantic + exact), row_variant="page_reorder")


def _question_reorder(row: DecisionIR) -> DecisionIR:
    term_dicts = list(row.metadata["terms"])
    if len(term_dicts) != 2:
        raise ValueError("question reorder is defined only for two-term compositions")
    reordered = list(reversed(term_dicts))
    question = "; also ".join(item["question"] for item in reordered)
    metadata = dict(row.metadata)
    metadata["question_order_reversed"] = True
    reordered_terms = tuple(
        Term(item["field_key"], item["question"], int(item["orientation"]), float(item["weight"]))
        for item in reordered
    )
    return _recalculate(row, variant="question_reorder", question=question, terms=reordered_terms, extra_metadata={"question_order_reversed": True})


def _candidate_count_probe(row: DecisionIR, world: World, candidate_count: int, prop_by_id: Mapping[str, Mapping[str, Any]]) -> DecisionIR:
    base_candidates = [candidate for candidate in row.candidates if candidate.id != UNKNOWN_ID]
    if candidate_count == 2:
        chosen = sorted(base_candidates, key=lambda c: hashlib.sha256(f"{row.id}:K2:{c.id}".encode()).hexdigest())[:2]
        keep_ids = {candidate.id for candidate in chosen}
        exact_ids = {_stable_id("exact", world.world_id, cid, length=18) for cid in keep_ids}
        blocks = tuple(block for block in row.state_blocks if (block.exact and block.id in exact_ids) or (not block.exact and row.metadata["page_owners"].get(block.id) in keep_ids))
        block_ids = {block.id for block in blocks}
        metadata = dict(row.metadata)
        for key in ("page_owners", "page_grades", "page_fields"):
            metadata[key] = {block_id: value for block_id, value in metadata[key].items() if block_id in block_ids}
        for key in ("known", "teacher_scores", "stable_ordinals", "exact_facts"):
            metadata[key] = {cid: value for cid, value in metadata[key].items() if cid in keep_ids or cid == UNKNOWN_ID}
        subrow = _replace_row(row, candidates=chosen + [next(c for c in row.candidates if c.id == UNKNOWN_ID)], blocks=blocks, metadata=metadata, gold=(UNKNOWN_ID,), row_variant="k2_source",
                             evidence=tuple(item for item in row.evidence if item.block_id in block_ids))
        return _recalculate(subrow, variant="probe_k2", extra_metadata={"probe_k": 2, "candidate_count": 2, "source_candidate_ids": sorted(keep_ids)})
    candidates = list(base_candidates)
    blocks = list(row.state_blocks)
    owners = dict(row.metadata["page_owners"])
    grades = dict(row.metadata["page_grades"])
    fields = dict(row.metadata["page_fields"])
    exact_facts = {key: dict(value) for key, value in row.metadata["exact_facts"].items() if key in {c.id for c in candidates}}
    stable_ordinals = {key: value for key, value in row.metadata["stable_ordinals"].items() if key in {c.id for c in candidates}}
    split_props = list(world.properties)
    query_prop_ids = {term["field_key"] for term in row.metadata["terms"]}
    if len(query_prop_ids) != 1:
        raise ValueError("candidate-count probe must use one atomic criterion")
    for candidate_index in range(4, candidate_count):
        cid = _stable_id("cand", world.world_id, "probe", candidate_index, length=12)
        candidates.append(Candidate(cid, "Alternative record."))
        stable_ordinals[cid] = candidate_index
        facts = _exact_values(world.world_id, cid)
        exact_facts[cid] = facts
        for prop in split_props:
            exposure = world.exposure_by_family[prop["family"]]
            grade = (4 * exposure + candidate_index) % 5
            variant = (world.local_index + candidate_index + FAMILIES.index(prop["family"])) % 3
            text = _page_wording(prop, world.split, grade, variant)
            block_id = _stable_id("page", world.world_id, cid, prop["property_id"], "primary", length=18)
            blocks.append(StateBlock(block_id, text, kind="semantic_page", exact=False))
            owners[block_id] = cid
            grades[block_id] = grade
            fields[block_id] = prop["property_id"]
        exact_id = _stable_id("exact", world.world_id, cid, length=18)
        blocks.append(StateBlock(exact_id, _exact_block_text(facts), kind="typed_exact_facts", exact=True))
    metadata = dict(row.metadata)
    metadata.update({
        "page_owners": owners, "page_grades": grades, "page_fields": fields,
        "exact_facts": exact_facts, "stable_ordinals": stable_ordinals,
    })
    expanded = _replace_row(row, candidates=candidates + [next(c for c in row.candidates if c.id == UNKNOWN_ID)], blocks=blocks, metadata=metadata, row_variant="candidate_count_source",
                            evidence=tuple(Evidence(block_id) for block_id in fields if fields[block_id] in {term["field_key"] for term in row.metadata["terms"]}))
    return _recalculate(expanded, variant=f"probe_k{candidate_count}", extra_metadata={"probe_k": candidate_count, "candidate_count": candidate_count, "source_candidate_ids": [c.id for c in candidates[:4]]})


def _query_from_metadata(row: DecisionIR) -> QuerySpec:
    terms = tuple(Term(item["field_key"], item["question"], int(item["orientation"]), float(item["weight"])) for item in row.metadata["terms"])
    return QuerySpec(str(row.metadata["query_id"]), str(row.metadata["query_kind"]), terms, row.question)


def _decision_ledger_row(row: DecisionIR, encoded: bytes) -> bytes:
    metadata = row.metadata
    value = {
        "decision_id": row.id,
        "row_sha256": _sha_bytes(encoded),
        "split": row.split,
        "world_id": metadata["world_id"],
        "variant": metadata["variant"],
        "query_id": metadata["query_id"],
        "query_kind": metadata["query_kind"],
        "family": metadata["family"],
        "parent_decision_id": metadata.get("parent_decision_id"),
        "gold_ids": list(row.gold),
    }
    return _canonical_json(value)


def _add_parent(row: DecisionIR, parent_id: str) -> DecisionIR:
    metadata = dict(row.metadata)
    metadata["parent_decision_id"] = parent_id
    return _replace_row(row, metadata=metadata, row_variant=str(metadata["variant"]))


def _validate_receipt(root: Path, source_sha: str, packet_sha: str, packet: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    if not receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError(f"accepted independent-audit receipt is absent or unsafe: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("receipt schema is absent or unrecognized")
    if receipt.get("status") != "accepted":
        raise ValueError("independent audit is not accepted")
    if receipt.get("source_sha256") != source_sha:
        raise ValueError("receipt is not bound to the current authored source hash")
    if receipt.get("packet_sha256") != packet_sha:
        raise ValueError("receipt is not bound to the current opaque review packet hash")
    reviewer_ids = receipt.get("reviewer_ids")
    if not isinstance(reviewer_ids, list) or len(reviewer_ids) < 2 or len(set(reviewer_ids)) != len(reviewer_ids) or any(not isinstance(value, str) or not value for value in reviewer_ids):
        raise ValueError("receipt must name at least two distinct reviewers")
    raw_relative = receipt.get("raw_reviews_file")
    if not isinstance(raw_relative, str) or not raw_relative:
        raise ValueError("receipt must identify preserved raw review records")
    raw_path = (root / raw_relative).resolve()
    if root.resolve() not in raw_path.parents or raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError("raw review file must be a regular file inside the private output root")
    raw_sha, _ = _sha_file(raw_path)
    if receipt.get("raw_reviews_sha256") != raw_sha:
        raise ValueError("raw review records do not match the receipt hash")
    rejected = receipt.get("rejected_review_ids", [])
    ambiguous = receipt.get("ambiguous_review_ids", [])
    if rejected or ambiguous:
        raise ValueError("receipt contains rejected or ambiguous review items")
    items = {item["review_id"] for item in packet["items"]}
    if receipt.get("item_count") != len(items):
        raise ValueError("receipt item_count does not cover the complete opaque review packet")
    raw_records: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"raw review line {line_no} is not an object")
            reviewer = record.get("reviewer_id")
            review_id = record.get("review_id")
            decision = record.get("decision")
            if reviewer not in reviewer_ids:
                raise ValueError(f"raw review line {line_no} names an unapproved reviewer")
            if review_id not in items:
                raise ValueError(f"raw review line {line_no} names an unknown opaque item")
            if decision != "accept":
                raise ValueError(f"raw review line {line_no} is not an unambiguous acceptance")
            raw_records.append(record)
    coverage: dict[str, set[str]] = {reviewer: set() for reviewer in reviewer_ids}
    for record in raw_records:
        reviewer = record["reviewer_id"]
        review_id = record["review_id"]
        if review_id in coverage[reviewer]:
            raise ValueError(f"duplicate raw review for item {review_id} by reviewer {reviewer}")
        coverage[reviewer].add(review_id)
    if any(reviewed != items for reviewed in coverage.values()):
        raise ValueError("every named reviewer must independently review every opaque packet item")
    if len(raw_records) != len(items) * len(reviewer_ids):
        raise ValueError("raw review count does not equal complete item-by-reviewer coverage")
    return {
        "receipt_sha256": _sha_file(receipt_path)[0],
        "raw_reviews_sha256": raw_sha,
        "reviewer_ids": reviewer_ids,
        "item_count": len(items),
        "raw_review_count": len(raw_records),
    }


def _load_prepared(root: Path, protocol_raw: bytes) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_dir = root / "source"
    audit_dir = root / "audit"
    manifest_path = source_dir / "prepare_manifest_v1.json"
    source_path = source_dir / "eca_authored_source_v1.json"
    inventory_path = source_dir / "eca_source_inventory_v1.json"
    packet_path = audit_dir / "opaque_review_packet_v1.json"
    key_path = audit_dir / "sealed_target_key_v1.json"
    for path in (manifest_path, source_path, inventory_path, packet_path, key_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"prepared source artifact is missing or unsafe: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    source_sha, _ = _sha_file(source_path)
    packet_sha, _ = _sha_file(packet_path)
    key_sha, _ = _sha_file(key_path)
    code_sha, _ = _sha_file(Path(__file__).resolve())
    protocol_sha = _sha_bytes(protocol_raw)
    if manifest.get("schema") != "vey.eca1.source-preparation-manifest.v1":
        raise ValueError("source preparation manifest schema mismatch")
    expected_file_hashes = {
        "source/eca_authored_source_v1.json": source_sha,
        "source/eca_source_inventory_v1.json": _sha_file(inventory_path)[0],
        "audit/opaque_review_packet_v1.json": packet_sha,
        "audit/sealed_target_key_v1.json": key_sha,
    }
    for relative, digest in expected_file_hashes.items():
        if manifest.get("files", {}).get(relative, {}).get("sha256") != digest:
            raise ValueError(f"prepared file hash mismatch: {relative}")
    if manifest.get("source_sha256") != source_sha:
        raise ValueError("source hash differs from preparation manifest")
    if manifest.get("builder_sha256") != code_sha:
        raise ValueError("builder code changed after source preparation; preserve lineage and prepare a new source version")
    if manifest.get("protocol_sha256") != protocol_sha:
        raise ValueError("protocol changed after source preparation")
    dependency_hashes = {relative: _sha_file(CANONICAL_ROOT / relative)[0] for relative in EXPECTED_CANONICAL_FILES}
    if manifest.get("canonical_dependency_sha256") != dependency_hashes:
        raise ValueError("canonical dependency hashes differ from source preparation")
    if source.get("schema") != SOURCE_SCHEMA or inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("source or inventory schema mismatch")
    if packet.get("schema") != PACKET_SCHEMA or key.get("schema") != KEY_SCHEMA:
        raise ValueError("opaque packet or sealed-key schema mismatch")
    if source_sha != inventory.get("source_sha256") or source_sha != packet.get("source_sha256") or source_sha != key.get("source_sha256"):
        raise ValueError("source, inventory, packet, and sealed key do not share one source hash")
    if packet_sha != key.get("packet_sha256"):
        raise ValueError("sealed target key does not bind the exact review packet")
    packet_ids = {item["review_id"] for item in packet.get("items", [])}
    key_ids = set(key.get("items", {}))
    if len(packet_ids) != len(packet.get("items", [])) or packet_ids != key_ids:
        raise ValueError("opaque audit IDs must map one-to-one and completely to the separate sealed key")
    if _sha_bytes(protocol_raw) != protocol_sha:
        raise ValueError("unreachable protocol hash inconsistency")
    return source, inventory, packet, {"manifest": manifest, "source_sha256": source_sha, "packet_sha256": packet_sha, "key_sha256": key_sha}


def _stream_build(root: Path, source: Mapping[str, Any], inventory: Mapping[str, Any], packet: Mapping[str, Any], source_lineage: Mapping[str, Any], protocol_raw: bytes, receipt_path: Path) -> dict[str, Any]:
    # Receipt validation is deliberately completed before creating the corpus
    # directory or any JSONL row file.
    audit_lineage = _validate_receipt(root, source_lineage["source_sha256"], source_lineage["packet_sha256"], packet, receipt_path)
    corpus_dir = root / "corpus"
    _private_directory(corpus_dir, create=True)
    output_paths = {split: corpus_dir / f"{split}.jsonl" for split in SPLITS}
    if any(path.exists() or path.is_symlink() for path in output_paths.values()):
        raise FileExistsError("one or more corpus split outputs already exist; refusing to overwrite")
    ledger_path = corpus_dir / "decision_ledger.jsonl"
    worlds_path = corpus_dir / "world_ledger.jsonl"
    manifest_path = corpus_dir / "build_manifest_v1.json"
    if any(path.exists() or path.is_symlink() for path in (ledger_path, worlds_path, manifest_path)):
        raise FileExistsError("a corpus ledger or manifest already exists; refusing to overwrite")

    source_props, properties_by_family = _property_maps(source)
    worlds = _world_plan(source)
    writers: dict[str, Any] = {}
    split_hashes: dict[str, Any] = {split: hashlib.sha256() for split in SPLITS}
    split_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    split_question_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    split_page_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    grade_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    family_world_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    row_ids: set[str] = set()
    row_ledger_hash = hashlib.sha256()
    world_ledger_hash = hashlib.sha256()
    row_count = 0
    try:
        for split, path in output_paths.items():
            writers[split] = path.open("xb")
        ledger_stream = ledger_path.open("xb")
        worlds_stream = worlds_path.open("xb")
        for world in worlds:
            families = list(world.families)
            for family in families:
                family_world_counts[world.split][family] += 1
            base_candidates, base_blocks, base_owners, base_grades, base_fields, exact_facts, stable_ordinals = _build_world_state(world, world.properties)
            for grade in base_grades.values():
                grade_counts[world.split][str(grade)] += 1
            world_record = {
                "world_id": world.world_id,
                "split": world.split,
                "split_index": world.local_index,
                "global_hash_rank": world.rank,
                "families": families,
                "field_keys": [prop["property_id"] for prop in world.properties],
                "field_codes": [prop["field_code"] for prop in world.properties],
                "exposure_index_by_family": dict(world.exposure_by_family),
            }
            world_line = _canonical_json(world_record)
            worlds_stream.write(world_line)
            world_ledger_hash.update(world_line)
            specs, _ = _query_specs(world, source_props, properties_by_family)
            present_keys = {prop["property_id"] for prop in world.properties}
            for spec_index, spec in enumerate(specs):
                base = _make_row(world, spec, base_candidates, base_blocks, base_owners, base_grades, base_fields, exact_facts, stable_ordinals, variant="base")
                base = _add_parent(base, base.id)
                base_id = base.id
                _emit_row(base, writers[world.split], ledger_stream, split_hashes, split_counts, variant_counts, split_question_hashes, split_page_hashes, row_ids, row_ledger_hash)
                row_count += 1
                for term_index, term in enumerate(spec.terms):
                    if term.property_id not in present_keys:
                        continue
                    for derived in (
                        _erase_variant(base, term.property_id),
                        _contradiction_variant(base, term.property_id, world.split, source_props),
                        _grade_change_variant(base, term_index, source_props, world.split),
                    ):
                        derived = _add_parent(derived, base_id)
                        _emit_row(derived, writers[world.split], ledger_stream, split_hashes, split_counts, variant_counts, split_question_hashes, split_page_hashes, row_ids, row_ledger_hash)
                        row_count += 1
                for permutation_index in (1, 2):
                    derived = _add_parent(_candidate_permutation(base, permutation_index), base_id)
                    _emit_row(derived, writers[world.split], ledger_stream, split_hashes, split_counts, variant_counts, split_question_hashes, split_page_hashes, row_ids, row_ledger_hash)
                    row_count += 1
                derived = _add_parent(_rename_candidates(base), base_id)
                _emit_row(derived, writers[world.split], ledger_stream, split_hashes, split_counts, variant_counts, split_question_hashes, split_page_hashes, row_ids, row_ledger_hash)
                row_count += 1
                derived = _add_parent(_page_reorder(base), base_id)
                _emit_row(derived, writers[world.split], ledger_stream, split_hashes, split_counts, variant_counts, split_question_hashes, split_page_hashes, row_ids, row_ledger_hash)
                row_count += 1
                if len(spec.terms) == 2:
                    derived = _add_parent(_question_reorder(base), base_id)
                    _emit_row(derived, writers[world.split], ledger_stream, split_hashes, split_counts, variant_counts, split_question_hashes, split_page_hashes, row_ids, row_ledger_hash)
                    row_count += 1
                if spec_index == 0:
                    for k in (2, 8, 16, 32, 64):
                        probe = _add_parent(_candidate_count_probe(base, world, k, source_props), base_id)
                        _emit_row(probe, writers[world.split], ledger_stream, split_hashes, split_counts, variant_counts, split_question_hashes, split_page_hashes, row_ids, row_ledger_hash)
                        row_count += 1
        for stream in writers.values():
            stream.flush()
            os.fsync(stream.fileno())
        ledger_stream.flush()
        os.fsync(ledger_stream.fileno())
        worlds_stream.flush()
        os.fsync(worlds_stream.fileno())
    finally:
        for stream in writers.values():
            stream.close()
        if "ledger_stream" in locals():
            ledger_stream.close()
        if "worlds_stream" in locals():
            worlds_stream.close()
    for split in SPLITS:
        if split_question_hashes[split].intersection(set().union(*(split_question_hashes[other] for other in SPLITS if other != split))):
            raise ValueError(f"generated semantic questions overlap across split: {split}")
        if split_page_hashes[split].intersection(set().union(*(split_page_hashes[other] for other in SPLITS if other != split))):
            raise ValueError(f"generated semantic pages overlap across split: {split}")
    if sum(split_counts.values()) != row_count or row_count != len(row_ids):
        raise ValueError("row count or decision ID uniqueness invariant failed")
    code_sha, code_size = _sha_file(Path(__file__).resolve())
    protocol_sha = _sha_bytes(protocol_raw)
    dependency_hashes = {relative: _sha_file(CANONICAL_ROOT / relative)[0] for relative in EXPECTED_CANONICAL_FILES}
    manifest = {
        "schema": BUILD_SCHEMA,
        "source_sha256": source_lineage["source_sha256"],
        "source_inventory_sha256": _sha_file(root / "source" / "eca_source_inventory_v1.json")[0],
        "opaque_packet_sha256": source_lineage["packet_sha256"],
        "sealed_key_sha256": source_lineage["key_sha256"],
        "receipt": audit_lineage,
        "builder_sha256": code_sha,
        "builder_bytes": code_size,
        "protocol_sha256": protocol_sha,
        "canonical_dependency_sha256": dependency_hashes,
        "seed": SEED,
        "grade_order": {"integer_values": list(GRADE_ORDER), "normalized_values": [0.0, 0.25, 0.5, 0.75, 1.0]},
        "world_counts": dict(WORLD_COUNTS),
        "world_count": len(worlds),
        "row_count": row_count,
        "row_counts_by_split": dict(split_counts),
        "variant_counts": dict(variant_counts),
        "family_world_counts_by_split": {split: dict(sorted(counter.items())) for split, counter in family_world_counts.items()},
        "page_grade_counts_by_split": {split: dict(sorted(counter.items())) for split, counter in grade_counts.items()},
        "decision_ids_sha256": row_ledger_hash.hexdigest(),
        "world_ledger_sha256": world_ledger_hash.hexdigest(),
        "split_jsonl_sha256": {split: split_hashes[split].hexdigest() for split in SPLITS},
        "semantic_text_disjointness": {
            "source_page_inventory_overlap": 0,
            "source_question_inventory_overlap": 0,
            "generated_question_hash_overlap": 0,
            "generated_page_hash_overlap": 0,
            "hash_normalization": "Unicode text is case-folded and whitespace-collapsed before SHA-256.",
        },
        "robustness_coverage": {
            "candidate_permutations_per_main_query": 2,
            "candidate_rename_per_main_query": 1,
            "page_reorder_per_main_query": 1,
            "question_reorder_per_composition": 1,
            "candidate_count_probes": [2, 8, 16, 32, 64],
            "candidate_count_scope": "one deterministic positive atomic criterion per world; probes are clustered variants, not independent worlds",
        },
        "exact_blocks": {
            "per_candidate": ["setup_charge_cents", "handoff_minutes", "replacement_parts"],
            "block_exact": True,
            "neural_input": False,
            "purpose": "mixed typed exact facts coexist with semantic pages and remain available only through exact blocks/metadata",
        },
        "outputs": {
            f"corpus/{split}.jsonl": {"rows": split_counts[split], "sha256": split_hashes[split].hexdigest()}
            for split in SPLITS
        } | {
            "corpus/decision_ledger.jsonl": {"rows": row_count, "sha256": row_ledger_hash.hexdigest()},
            "corpus/world_ledger.jsonl": {"rows": len(worlds), "sha256": world_ledger_hash.hexdigest()},
        },
    }
    manifest_bytes = _canonical_json(manifest)
    _write_exclusive(manifest_path, manifest_bytes)
    return manifest


def _emit_row(row: DecisionIR, stream: Any, ledger_stream: Any, split_hashes: Mapping[str, Any], split_counts: Counter[str], variant_counts: Counter[str], split_question_hashes: Mapping[str, set[str]], split_page_hashes: Mapping[str, set[str]], row_ids: set[str], row_ledger_hash: Any) -> None:
    if row.id in row_ids:
        raise ValueError(f"duplicate decision ID: {row.id}")
    row_ids.add(row.id)
    encoded = (row.to_json() + "\n").encode("utf-8")
    stream.write(encoded)
    split_hashes[row.split].update(encoded)
    split_counts[row.split] += 1
    variant_counts[str(row.metadata["variant"])] += 1
    split_question_hashes[row.split].add(_sha_bytes(_normal_text(row.question).encode("utf-8")))
    for block in row.state_blocks:
        if not block.exact:
            split_page_hashes[row.split].add(_sha_bytes(_normal_text(block.text).encode("utf-8")))
    ledger_line = _decision_ledger_row(row, encoded)
    ledger_stream.write(ledger_line)
    row_ledger_hash.update(ledger_line)



def _export_authorship_seed(root: Path) -> dict[str, Any]:
    source_dir = root / "source"
    _private_directory(root, create=True)
    _private_directory(source_dir, create=True)
    seed = {
        "schema": "vey.eca1.authorship-seed.v1",
        "seed": SEED,
        "families": list(FAMILIES),
        "fields": [
            {
                "family": family, "field_code": field_code, "title": title,
                "criterion": criterion, "description": description, "grade_rubrics": list(clauses),
            }
            for family, field_code, title, criterion, description, clauses in AUTHORED_FIELDS
        ],
        "page_templates": {split: list(templates) for split, templates in PAGE_TEMPLATES.items()},
        "question_templates": {
            split: {str(orientation): list(templates) for orientation, templates in orientations.items()}
            for split, orientations in QUESTION_TEMPLATES.items()
        },
        "exact_fact_block_template": "Quoted setup charge {setup_charge_cents} cents; handoff delay {handoff_minutes} minutes; replacement parts {replacement_parts} units.",
        "composition_separator": "; also ",
        "review_instructions": "For each opaque item, assess whether the page's stated extent fits the ordered neutral rubric, or whether the question clearly asks about its described property. Write one raw JSONL review object per review_id with reviewer_id, review_id, decision (accept, ambiguous, or reject), and optional note. Do not infer unstated facts.",
    }
    payload = _canonical_json(seed)
    digest = _write_exclusive(source_dir / "eca_authorship_seed_v1.json", payload)
    return {"seed_sha256": digest, "seed_bytes": len(payload)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--export-authorship-seed", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--prepare-source", action="store_true", help="write immutable private authored source, audit packet, and sealed key")
    modes.add_argument("--build", action="store_true", help="stream canonical DecisionIR JSONL after independent audit acceptance")
    parser.add_argument("--receipt", type=Path, help="accepted audit receipt; default is output_root/audit/accepted_receipt_v1.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    protocol, protocol_raw, root = _protocol()
    if args.export_authorship_seed:
        result = _export_authorship_seed(root)
        print(json.dumps({"status": "private_authorship_seed_written", **result}, sort_keys=True))
        return 0
    if args.prepare_source:
        result = _prepare_source(root, protocol, protocol_raw)
        print(json.dumps({
            "status": "source_prepared_for_independent_review",
            "output_root": str(root),
            "source_sha256": result["source_sha256"],
            "audit_item_count": result["audit_item_count"],
            "audit_items_by_type": result["audit_items_by_type"],
            "source_coverage": result["source_coverage"],
        }, sort_keys=True))
        return 0
    _private_directory(root, create=False)
    source, inventory, packet, lineage = _load_prepared(root, protocol_raw)
    receipt_path = args.receipt or (root / "audit" / "accepted_receipt_v1.json")
    result = _stream_build(root, source, inventory, packet, lineage, protocol_raw, receipt_path)
    print(json.dumps({
        "status": "corpus_built_after_accepted_independent_review",
        "output_root": str(root),
        "world_count": result["world_count"],
        "row_count": result["row_count"],
        "row_counts_by_split": result["row_counts_by_split"],
        "source_sha256": result["source_sha256"],
        "manifest": str(root / "corpus" / "build_manifest_v1.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError, OSError) as exc:
        print(f"ECA build blocked: {exc}", file=sys.stderr)
        raise SystemExit(2)
