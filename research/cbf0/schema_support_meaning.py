"""Combine blind semantic judgments; exact syntax, never raters, owns weights."""
import argparse
import hashlib
import json
from pathlib import Path

from audit import put
from build import canonical
from schema_grounding_compiler import parse_criterion


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assemble(root, judgment_paths):
    root = Path(root)
    manifest = load(root / 'corpus_manifest.json')
    packet_path = root / 'blind_meaning_questions.json'
    assert sha(packet_path) == manifest['blind_review']['packet_sha256']
    packet = load(packet_path)
    texts = {q['review_id']: q['text'] for q in packet['questions']}
    rows, sources = [], []
    for path in judgment_paths:
        value = load(path)
        assert set(value) == {'schema_version', 'packet_sha256', 'judgments'}
        assert value['schema_version'] == packet['schema_version']
        assert value['packet_sha256'] == sha(packet_path)
        rows.extend(value['judgments'])
        sources.append(dict(path=str(Path(path)), sha256=sha(path), rows=len(value['judgments'])))
    assert len(rows) == len({r['review_id'] for r in rows}) == len(texts) == 352
    assert {r['review_id'] for r in rows} == set(texts)
    rendered, vector_differences = [], []
    for raw in rows:
        assert raw['decision'] in ('accept', 'ambiguous', 'reject')
        if raw['decision'] != 'accept':
            rendered.append(raw)
            continue
        terms = parse_criterion(texts[raw['review_id']])
        if raw['kind'] == 'atomic':
            assert set(raw) == {'review_id', 'decision', 'kind', 'axis', 'sign'}
            assert type(raw['axis']) is int and 0 <= raw['axis'] < 4
            assert type(raw['sign']) is int and raw['sign'] in (-1, 1)
            assert len(terms) == 1 and terms[0]['factor'] == 1
            rendered.append(raw)
        else:
            assert raw['kind'] == 'composition'
            assert set(raw) == {'review_id', 'decision', 'kind', 'components', 'weights'}
            assert len(terms) == len(raw['components']) == 2
            weights = [0] * 4
            for term, semantic in zip(terms, raw['components']):
                assert set(semantic) == {'axis', 'sign'}
                assert type(semantic['axis']) is int and 0 <= semantic['axis'] < 4
                assert type(semantic['sign']) is int and semantic['sign'] in (-1, 1)
                assert term['literal_axis'] is None and type(term['factor']) is int
                weights[semantic['axis']] += term['factor'] * semantic['sign']
            assert len(raw['weights']) == 4 and all(type(v) is int for v in raw['weights'])
            if raw['weights'] != weights:
                vector_differences.append(dict(review_id=raw['review_id'], raw_weights=raw['weights'], exact_weights=weights))
            rendered.append(dict(raw, weights=weights))
    output = dict(schema_version=packet['schema_version'], packet_sha256=sha(packet_path), judgments=rendered)
    predictions_name = manifest['blind_review']['predictions_file']
    put(root, predictions_name, canonical(output) + b'\n')
    key = load(root / 'blind_meaning_key.json')
    issues = []
    for row in rendered:
        expected = key[row['review_id']]['expected']
        accepted = row['decision'] == 'accept'
        if accepted and 'axis' in expected:
            accepted = row['kind'] == 'atomic' and (row['axis'], row['sign']) == (expected['axis'], expected['sign'])
        elif accepted:
            components = [dict(axis=c['axis'], sign=c['sign']) for c in expected['components']]
            accepted = row['kind'] == 'composition' and row['components'] == components and row['weights'] == expected['weights']
        if not accepted:
            issues.append(dict(review_id=row['review_id'], decision=row['decision'], cohort=key[row['review_id']]['cohort'], record_id=key[row['review_id']]['record_id']))
    audit = dict(all_meanings_accepted=not issues, evidence_class='MEASURED: automated independent blind semantic review; not human annotation', review_rows=352,
                 final_atoms_sha256=manifest['files']['final_atoms.json'], final_compositions_sha256=manifest['files']['final_compositions.json'],
                 blind_predictions_sha256=sha(root / predictions_name), raw_judgment_sources=sources,
                 raw_vector_differences=vector_differences, blocking_rows=issues,
                 exact_weight_authority='Frozen parse_criterion factors combined with independently judged per-term axis/sign; sealed key never renders weights.',
                 assembler_sha256=sha(Path(__file__)))
    put(root, 'meaning_audit.json', canonical(audit) + b'\n')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('judgments', nargs='+')
    args = parser.parse_args()
    print(json.dumps(assemble(args.root, args.judgments), indent=2, sort_keys=True))
