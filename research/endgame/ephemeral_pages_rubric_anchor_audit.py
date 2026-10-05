#!/usr/bin/env python3
"""Stage S0 mechanical screens for per-property rubric anchors.

Builds one anchor per authored property by concatenating that property's ordered
grade_rubrics stage strings in grade_order, then runs the mechanical screens
declared by the property-definition audit prerequisite of
research/endgame/ephemeral_pages_grade_interface_preregistration.json.

This is a mechanical screen only. A mechanical pass is not audit acceptance.
The blocking two-reviewer attestation is declared in
research/endgame/ephemeral_pages_rubric_anchor_audit_protocol.json and is not
performed here. Authored rubric, question, page and property description text is
private property data: the receipt and stdout carry counts, digests and
per-condition booleans only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

DEFAULT_SOURCE = Path('/home/fazinahamed/Documents/vey-data/decisionmix/endgame/'
                      'ephemeral-pages-v1/source/eca_authored_source_v1.json')

# Never opened by this tool. The untouched ECA-2 final stays sealed; refusing by
# path and by component name keeps a wrong --source from decoding final material.
FORBIDDEN_SOURCE_SUBSTRINGS = ('ephemeral-pages-atomic-v2', '/final.', 'final_manifest',
                               'final_records', 'final.jsonl')
FORBIDDEN_SOURCE_BASENAMES = frozenset({
    'eca2_final_source.json',
    'eca2_final_inventory.json',
    'eca2_final_prepare_manifest.json',
    'eca2_final_source_authorship_receipt.json',
    'final.jsonl',
    'final_records.jsonl',
    'final_manifest.json',
})

# Normalized token Jaccard ceiling for screen 7.
JACCARD_LIMIT = 0.60

# Screen 3 vocabulary: Unicode digits, matched as digit runs on casefolded text.
DIGIT_RUN = re.compile(r'\d+')

# Screen 5 vocabulary: stage headings and ordinal markers. Closed set, matched on
# NFC casefolded text with word boundaries. Grounded in the marker words that
# actually occur in authored rubric text (most, stage, stages, step, steps) plus
# the ordinal position words and scale nouns the preregistration names.

STAGE_MARKER_PATTERN = re.compile(
    r'(?<![0-9a-z])(?:'
    r'first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|'
    r'last|final|lowest|highest|least|most|'
    r'stage|stages|step|steps|tier|tiers|level|levels|band|bands|'
    r'grade|grades|score|scores|point|points|rating|ratings|'
    r'ordinal|scale|phase|phases'
    r')(?![0-9a-z])'
)
STAGE_MARKER_TOKENS = tuple(
    token for token in STAGE_MARKER_PATTERN.pattern.split('|')
    if token and token.isalpha()
)
WORD_RE = re.compile(r'[0-9a-z]+')


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize(text):
    """NFC, casefold, collapse all whitespace runs to one space, strip."""
    folded = unicodedata.normalize('NFC', text).casefold()
    return ' '.join(folded.split())


def tokens(normalized):
    return WORD_RE.findall(normalized)


def jaccard(left_tokens, right_tokens):
    """Token Jaccard on the multiset-free set view; 1.0 when both are empty."""
    left, right = set(left_tokens), set(right_tokens)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def refuse_final_source(path):
    resolved = Path(path)
    text = str(resolved)
    if resolved.name in FORBIDDEN_SOURCE_BASENAMES:
        raise SystemExit(f'refusing sealed final-pool source: {resolved.name}')
    for marker in FORBIDDEN_SOURCE_SUBSTRINGS:
        if marker in text:
            raise SystemExit(f'refusing sealed final-pool source path: {marker}')


def build_anchor(property_record):
    """Concatenate grade_rubrics in grade_order, no separators that encode stage
    boundaries, no ordinal markers and no headings introduced."""
    order = property_record['grade_order']
    rubrics = property_record['grade_rubrics']
    if sorted(order) != list(range(len(rubrics))):
        raise ValueError('grade_order is not a permutation of the rubric stage indices')
    stages = [rubrics[index] for index in order]
    return ' '.join(stage.strip() for stage in stages if stage.strip()), len(stages)


def screens_for_property(anchor, property_record):
    """Return the per-condition receipt rows for one property."""
    anchor_normalized = normalize(anchor)
    anchor_tokens = tokens(anchor_normalized)
    grade_order = property_record['grade_order']
    comparable = ([str(question['text']) for question in property_record['questions']]
                  + [wording['text'] for wording in property_record['page_wordings']])

    grade_order_labels = sorted({str(label) for label in grade_order})
    grade_label_hits = sorted({label for label in grade_order_labels if label in anchor_normalized})

    marker_hits = sorted({token for token in STAGE_MARKER_TOKENS
                          if re.search(rf'(?<![0-9a-z]){re.escape(token)}(?![0-9a-z])',
                                       anchor_normalized)})

    equality_hits, containment_hits, jaccard_pairs = 0, 0, []
    for text in comparable:
        text_normalized = normalize(text)
        text_tokens = tokens(text_normalized)
        if anchor_normalized == text_normalized:
            equality_hits += 1
            continue
        if anchor_normalized in text_normalized or text_normalized in anchor_normalized:
            containment_hits += 1
        score = jaccard(anchor_tokens, text_tokens)
        if score >= JACCARD_LIMIT:
            jaccard_pairs.append(score)

    return [
        {'condition': 'screen_1_one_anchor_per_property_and_count_equality',
         'passed': True,
         'detail': 'anchor count equals property count; one anchor built per property id'},
        {'condition': 'screen_2_anchor_nonempty',
         'passed': bool(anchor_normalized),
         'detail': {'anchor_char_count': len(anchor), 'anchor_normalized_char_count': len(anchor_normalized)}},
        {'condition': 'screen_3_no_unicode_digit_in_anchor',
         'passed': DIGIT_RUN.search(anchor_normalized) is None,
         'detail': {'digit_run_count': len(DIGIT_RUN.findall(anchor_normalized))}},
        {'condition': 'screen_4_no_grade_order_label_in_anchor',
         'passed': not grade_label_hits,
         'detail': {'grade_order_label_count': len(grade_order_labels), 'hit_labels': grade_label_hits}},
        {'condition': 'screen_5_no_stage_heading_or_ordinal_marker_in_anchor',
         'passed': not marker_hits,
         'detail': {'marker_vocabulary_size': len(STAGE_MARKER_TOKENS), 'hit_marker_tokens': marker_hits}},
        {'condition': 'screen_6_anchor_distinct_from_question_and_page_text',
         'passed': equality_hits == 0 and containment_hits == 0,
         'detail': {'equal_comparisons': equality_hits, 'containment_comparisons': containment_hits,
                    'comparisons': len(comparable)}},
        {'condition': 'screen_7_token_jaccard_below_limit',
         'passed': not jaccard_pairs,
         'detail': {'jaccard_limit': JACCARD_LIMIT, 'comparisons': len(comparable),
                    'pairs_at_or_above_limit': len(jaccard_pairs),
                    'max_jaccard': round(max((jaccard(anchor_tokens, tokens(normalize(text)))
                                             for text in comparable), default=0.0), 6)}},
    ]



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                        help='authored prefinal source JSON')
    parser.add_argument('--output', type=Path, required=True,
                        help='exclusive-create receipt path')
    args = parser.parse_args()

    refuse_final_source(args.source)
    source_sha256 = sha256_file(args.source)
    source = json.loads(args.source.read_text())

    properties = source['properties']
    anchors, rows, counts = {}, [], {'properties': 0, 'stages': 0}
    for record in properties:
        property_id = record['property_id']
        anchor, stage_count = build_anchor(record)
        anchors[property_id] = anchor
        counts['properties'] += 1
        counts['stages'] += stage_count
        conditions = screens_for_property(anchor, record)
        rows.append({'property_id': property_id,
                     'anchor_sha256': sha256_text(anchor),
                     'anchor_char_count': len(anchor),
                     'anchor_normalized_char_count': len(normalize(anchor)),
                     'stage_count': stage_count,
                     'grade_order': record['grade_order'],
                     'passed': all(condition['passed'] for condition in conditions),
                     'conditions': conditions})

    distinct_anchors = {digest for digest in
                        (sha256_text(anchor) for anchor in anchors.values())}
    screen_failures = {}
    for row in rows:
        for condition in row['conditions']:
            if not condition['passed']:
                screen_failures.setdefault(condition['condition'], []).append(row['property_id'])
    passed_properties = sum(1 for row in rows if row['passed'])

    result = {
        'schema': 'vey.eca.rubric-anchor-audit-receipt.v1',
        'stage': 'S0_custody mechanical screens only',
        'audit_prerequisite_source': 'research/endgame/'
                                     'ephemeral_pages_grade_interface_preregistration.json',
        'authored_source': {'path': str(args.source), 'sha256': source_sha256},
        'anchor_rule': 'concatenate grade_rubrics in grade_order, single space, '
                       'no ordinal markers, no stage labels, no boundary-encoding separators',
        'normalization': 'NFC, casefold, collapse whitespace, strip',
        'jaccard_limit': JACCARD_LIMIT,
        'counts': {
            'properties': counts['properties'],
            'anchors_built': len(anchors),
            'rubric_stage_strings': counts['stages'],
            'distinct_anchor_sha256': len(distinct_anchors),
            'properties_passed_all_screens': passed_properties,
            'properties_failed_any_screen': counts['properties'] - passed_properties,
            'questions_compared': sum(len(record['questions']) for record in properties),
            'page_wording_comparisons': sum(len(record['page_wordings']) for record in properties),
        },
        'screen_failures_by_condition': {condition: sorted(ids)
                                         for condition, ids in sorted(screen_failures.items())},
        'mechanical_pass': passed_properties == counts['properties'] == len(anchors),
        'audit_acceptance': False,
        'audit_acceptance_note': 'A mechanical pass is not audit acceptance. The blocking '
                                 'two-reviewer attestation is declared in '
                                 'ephemeral_pages_rubric_anchor_audit_protocol.json and has '
                                 'not been performed.',
        'properties': rows,
        'neural_encoder_forwards': 0,
        'model_training': False,
        'fitting_performed': False,
        'network_access': False,
        'final_pool_access': False,
        'final_opened': False,
        'privacy': 'Anchor, rubric, question, page and property description text is private '
                   'property data. This receipt records per-property digests, counts and '
                   'per-condition booleans only.',
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

    print(json.dumps({'schema': result['schema'], 'status': 'mechanical_pass'
                      if result['mechanical_pass'] else 'mechanical_screen_failure',
                      'authored_source_sha256': source_sha256, 'counts': result['counts'],
                      'screen_failures_by_condition': {k: len(v) for k, v
                                                        in result['screen_failures_by_condition'].items()},
                      'audit_acceptance': False, 'final_opened': False,
                      'output': str(args.output), 'output_sha256': sha256_file(args.output)},
                     sort_keys=True))


if __name__ == '__main__':
    main()