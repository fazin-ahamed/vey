"""Freeze CBF-8 source copies and generated cohorts before model execution."""
import argparse
import hashlib
import json
import re
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from audit import put
from build import canonical
from schema_grounding_compiler import compile_criterion, parse_criterion

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'schema_support_protocol.json'
CBF7_PROTOCOL = HERE / 'relation_pretrained_protocol.json'
CBF7_RESULTS = HERE / 'relation_pretrained_result_manifest.json'
CBF6_RESULTS = HERE / 'schema_relation_result_manifest.json'
COPY_FILES = (
    'training_atoms.json', 'validation_atoms.json', 'literal_holdout_atoms.json',
    'development_atoms.json', 'development_compositions.json', 'literal_cases.json', 'states.json',
)
REFERENCE_FILES = {
    'development_atoms.json': 32, 'development_compositions.json': 50,
    'final_atoms.json': 128, 'final_compositions.json': 128,
    'training_atoms.json': 32, 'validation_atoms.json': 32,
    'literal_holdout_atoms.json': 32, 'literal_cases.json': 155,
}
BASELINE_FILES = {
    'checkpoint': 'nli_xsmall_higher_cls_checkpoint.pt',
    'training': 'nli_xsmall_higher_cls_training.json',
    'normalizer': 'normalizer_nli_xsmall_higher_cls.npz',
    'features': 'features_nli_xsmall_higher_cls.npy',
    'inputs': 'inputs_nli_xsmall_higher.json',
    'lineage': 'encoder_lineage_nli_xsmall.json',
}
EXPECTED_DEFINITIONS = {
    'reliability': 'The proportion of matched-use trials in which the item performs its intended task without a failure. A higher value means fewer failed trials at equal exposure.',
    'purchase expense': 'The money paid to acquire the item at the initial transaction, excluding later payments for using or maintaining it. A higher value means a larger initial payment.',
    'operating expense': 'The money paid after acquisition to keep using or maintaining the item over the same use period, excluding the initial transaction. A higher value means larger ongoing payments.',
    'convenience': 'Ease of preparing and using the item for the same intended task. A higher value means less user effort, fewer required actions, or less setup burden. Monetary payments are separate attributes.',
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def tokens(text):
    return tuple(re.findall(r'[a-z0-9]+', text.casefold()))


def contains(text_tokens, phrase_tokens):
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    return any(text_tokens[i:i + len(phrase_tokens)] == phrase_tokens
               for i in range(len(text_tokens) - len(phrase_tokens) + 1))


def normalized(text):
    return ' '.join(tokens(text))


def semantic_phrases(text):
    return [text[t['span'][0]:t['span'][1]] for t in parse_criterion(text)]


def _check_c7_source(cfg):
    source = Path(cfg['source_root'])
    c7_cfg = load(CBF7_PROTOCOL)
    c7_results = load(CBF7_RESULTS)
    result_bytes = CBF7_RESULTS.read_bytes()
    c7_corpus = load(source / 'corpus_manifest.json')
    c7_verification_path = source / 'verification.json'
    verification_bytes = c7_verification_path.read_bytes()
    verification = json.loads(verification_bytes)

    assert c7_cfg['output_root'] == str(source)
    assert c7_results['output_root'] == str(source)
    assert c7_results['all_independent_checks_pass'] is True
    assert c7_results['B_STEF_allowed'] is False
    assert c7_results['protocol_sha256'] == sha(CBF7_PROTOCOL)
    assert c7_results['verification_sha256'] == sha_bytes(verification_bytes)
    assert c7_results['results_sha256'] == sha(source / 'results.json')
    assert verification['all_checks_pass'] is True
    assert verification['baseline_byte_identical_reuse'] is True
    assert c7_corpus['experiment'] == 'CBF-7'
    assert c7_corpus['final_closed_until_selection'] is True
    assert c7_corpus['protocol_sha256'] == sha(CBF7_PROTOCOL)

    for name, expected in c7_corpus['files'].items():
        assert sha(source / name) == expected, f'CBF7 corpus artifact changed: {name}'
    c6_source = Path(c7_cfg['source_root'])
    for name, expected in c7_corpus['source_verified_files'].items():
        assert sha(c6_source / name) == expected, f'CBF7 source reference changed: {name}'
    checked = verification['checked_source_and_model_sha256']
    refs = {}
    for role, name in BASELINE_FILES.items():
        path = source / name
        if role == 'training':
            expected = sha(path)
            custody = 'CBF8 preparation-time metadata byte pin; CBF7 reconstructed head fields checked below'
        else:
            expected = checked.get(str(path))
            assert expected is not None, f'CBF7 independent verification omits baseline {name}'
            custody = 'CBF7 independent verification byte hash'
        assert sha(path) == expected, f'CBF7 baseline changed: {name}'
        refs[role] = dict(file=name, source_path=str(path), sha256=expected, custody_basis=custody)
    assert c7_results['baseline_byte_identical_reuse'] is True
    baseline_training = load(source / BASELINE_FILES['training'])
    assert baseline_training['arm_id'] == 'nli_xsmall_higher_cls'
    assert baseline_training['feature'] == 'CLS'
    assert baseline_training['frozen_encoder_before_sha256'] == baseline_training['frozen_encoder_after_sha256']
    assert baseline_training['first_gradient_L1'] > 0
    assert baseline_training['feature_sha256'] == refs['features']['sha256']
    assert baseline_training['evidence_sha256'] == refs['inputs']['sha256']
    assert baseline_training['normalizer']['arm_id'] == 'nli_xsmall_higher_cls'
    verified_head = verification['stages']['nli_xsmall_higher_cls']['head']
    assert baseline_training['selected_epoch'] == verified_head['selected_epoch']
    assert abs(baseline_training['validation_CE'] - verified_head['unweighted_validation_CE']) <= 2e-5
    assert baseline_training['frozen_encoder_before_sha256'] == verified_head['frozen_parameter_sha256']
    baseline_lineage = load(source / BASELINE_FILES['lineage'])
    c8_model = cfg['model']
    assert baseline_lineage['revision'] == c8_model['revision']
    assert baseline_lineage['weight_sha256'] == c8_model['weight_sha256']
    assert baseline_lineage['floating_parameters'] == c8_model['floating_parameters'] == 70831107
    assert baseline_lineage['width'] == c8_model['width'] == 384
    return dict(
        source_root=str(source), c7_protocol_sha256=sha(CBF7_PROTOCOL),
        source_result_manifest_sha256=sha_bytes(result_bytes),
        source_results_sha256=c7_results['results_sha256'],
        source_verification_sha256=sha_bytes(verification_bytes),
        source_corpus_manifest_sha256=sha(source / 'corpus_manifest.json'),
        source_measurement_git_revision=c7_results['measurement_git'],
        source_inherited_measurement_git_revision=c7_corpus['source_measurement_git_revision'],
        source_verified_files={name: sha(source / name) for name in COPY_FILES + tuple(BASELINE_FILES.values())},
        baseline_references=dict(arm='literal_generic', format='generic', feature='raw final-layer CLS',
                                no_retrain=True, no_reencode=True, files=refs),
        source_lineage=dict(c7_output_root=str(source), c7_protocol_sha256=sha(CBF7_PROTOCOL),
                            c7_results_sha256=c7_results['results_sha256'],
                            c7_verification_sha256=sha_bytes(verification_bytes),
                            c7_corpus_manifest_sha256=sha(source / 'corpus_manifest.json')),
    )


def _source_copy_and_reference_check(cfg, source_info):
    source = Path(cfg['source_root'])
    c6_cfg = load(CBF7_PROTOCOL)
    c6_root = Path(c6_cfg['source_root'])
    c6_inventory = load(CBF6_RESULTS)
    c7_corpus = load(source / 'corpus_manifest.json')
    for name, count in REFERENCE_FILES.items():
        c7_path = source / name
        rows = load(c7_path)
        assert len(rows) == count, f'CBF7 reference row count: {name}'
        c7_expected = c7_corpus['files'].get(name)
        assert c7_expected is not None and sha(c7_path) == c7_expected, f'CBF7 reference hash: {name}'
        c6_expected = c6_inventory['files'][name]['sha256']
        assert sha(c6_root / name) == c6_expected, f'CBF6 reference hash: {name}'
        if name in COPY_FILES:
            assert c7_path.read_bytes() == (c6_root / name).read_bytes(), f'CBF6 byte-copy mismatch: {name}'
    return c6_root, c6_inventory


def _references(cfg, c6_root, c6_inventory):
    c7_root = Path(cfg['source_root'])
    refs, lineage = [], {}
    for label, root in (('CBF6', c6_root), ('CBF7', c7_root)):
        source_inventory = c6_inventory['files'] if label == 'CBF6' else load(c7_root / 'corpus_manifest.json')['files']
        for name, expected_rows in REFERENCE_FILES.items():
            path = root / name
            expected = source_inventory[name]
            expected_sha = expected['sha256'] if isinstance(expected, dict) else expected
            payload = path.read_bytes()
            rows = json.loads(payload)
            assert sha_bytes(payload) == expected_sha and len(rows) == expected_rows, f'{label} lexical reference {name}'
            lineage[f'{label}/{name}'] = dict(path=str(path), sha256=expected_sha, rows=len(rows))
            refs.extend(dict(id=f'{label}:{row["id"]}', text=row['text'], source=f'{label}/{name}') for row in rows)
    return refs, lineage


def _check_groups(rows, expected_n, pairs, per_direction, label):
    assert len(rows) == expected_n, f'{label} count'
    assert len({row['id'] for row in rows}) == expected_n, f'{label} unique ids'
    assert Counter((row['axis'], row['sign']) for row in rows) == Counter(
        {(axis, sign): per_direction for axis in range(4) for sign in (-1, 1)}), f'{label} field/sign balance'
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['pair_id']].append(row)
    assert len(grouped) == pairs and all(len(group) == 2 for group in grouped.values()), f'{label} pair counts'
    for group in grouped.values():
        a, b = group
        assert a['axis'] == b['axis'] and a['sign'] == -b['sign']
        assert a['weights'] == [-weight for weight in b['weights']]
        assert a['text'] != b['text']
    return grouped


def _compile_checks(training_atoms, validation_atoms, final_atoms, compositions):
    all_atoms = training_atoms + validation_atoms + final_atoms
    by_phrase = {normalized(atom['text']): atom for atom in all_atoms}
    assert len(by_phrase) == len(all_atoms), 'atomic phrase collision across CBF8 cohorts'
    records = []
    uses = Counter()
    coverage = Counter()

    def compile_row(row, expected_calls):
        calls = []

        def teacher(term):
            lo, hi = term['context_span']
            phrase = normalized(row['text'][lo:hi].rstrip('.') + '.')
            atom = by_phrase[phrase]
            calls.append(atom['id'])
            return dict(axis=atom['axis'], sign=atom['sign'])

        vector, ir = compile_criterion(row['text'], teacher)
        assert vector == row['weights'], f'compiler weights mismatch {row["id"]}'
        parsed = parse_criterion(row['text'])
        assert len(parsed) == expected_calls and all(term['literal_axis'] is None for term in parsed), row['id']
        assert len(calls) == expected_calls, row['id']
        return vector, ir, parsed, calls

    for cohort_name, atoms in (('semantic_training', training_atoms),
                               ('semantic_validation', validation_atoms), ('final_atoms', final_atoms)):
        for row in atoms:
            vector, ir, parsed, calls = compile_row(row, 1)
            assert parsed[0]['factor'] == 1
            expected = [0] * 4
            expected[row['axis']] = row['sign']
            assert vector == expected
            records.append(dict(cohort=cohort_name, id=row['id'], weights=vector, atoms=ir, atomic_ids=calls))
    final_by_id = {row['id']: row for row in final_atoms}
    for row in compositions:
        vector, ir, parsed, calls = compile_row(row, 2)
        assert len(row['components']) == 2
        details = []
        for component, term, atom_id in zip(row['components'], parsed, calls):
            atom = final_by_id[component['atom_id']]
            lo, hi = term['context_span']
            assert atom_id == atom['id']
            assert normalized(row['text'][lo:hi].rstrip('.')) == normalized(atom['text'].rstrip('.'))
            assert component['factor'] == term['factor'] and component['atom_id'] == atom_id
            uses[atom_id] += 1
            details.append((atom['axis'], atom['sign'], component['factor']))
        coverage[tuple(details)] += 1
        records.append(dict(cohort='final_compositions', id=row['id'], weights=vector,
                            atoms=ir, atomic_ids=calls))
    assert set(uses) == set(final_by_id) and set(uses.values()) == {2}, 'each final atom must appear exactly twice'
    expected_coverage = {
        tuple(zip(axes, signs, factors))
        for axes in __import__('itertools').combinations(range(4), 2)
        for signs in __import__('itertools').product((-1, 1), repeat=2)
        for factors in __import__('itertools').product((1, 2), repeat=2)
    }
    assert set(coverage) == expected_coverage, 'composition support/sign/factor coverage'
    return records, dict(sorted(uses.items())), {repr(k): v for k, v in sorted(coverage.items())}


def _lexical_checks(all_new, training_atoms, validation_atoms, final_atoms, final_compositions, references):
    old_phrases = []
    for reference in references:
        old_phrases.append((reference['source'], reference['id'], 'full', reference['text']))
        for phrase in semantic_phrases(reference['text']):
            old_phrases.append((reference['source'], reference['id'], 'semantic', phrase))
    c8_training_phrases = [
        ('CBF8_training_validation', row['id'], kind, phrase)
        for row in training_atoms + validation_atoms
        for kind, phrase in [('full', row['text'])] + [('semantic', p) for p in semantic_phrases(row['text'])]
    ]
    final_ids = {row['id'] for row in final_atoms + final_compositions}
    fields = tuple(EXPECTED_DEFINITIONS)
    issues = []
    for row in all_new:
        row_tokens = tokens(row['text'])
        if any(contains(row_tokens, tokens(field)) for field in fields):
            issues.append(dict(id=row['id'], reason='canonical field-name leakage'))
        try:
            parsed = parse_criterion(row['text'])
        except ValueError as error:
            raise ValueError(f'compiler rejected new phrase {row["id"]}: {error}') from error
        if not parsed or any(term['literal_axis'] is not None for term in parsed):
            issues.append(dict(id=row['id'], reason='literal/full-field compiler dispatch'))
        phrase_scope = old_phrases + (c8_training_phrases if row['id'] in final_ids else [])
        for group, source_id, kind, phrase in phrase_scope:
            if contains(row_tokens, tokens(phrase)):
                issues.append(dict(id=row['id'], reference_group=group, source_id=source_id,
                                   phrase_kind=kind, phrase=phrase))
    assert not issues, f'lexical or full-field leakage: {issues[:8]}'

    texts = [normalized(row['text']) for row in all_new]
    assert len(texts) == len(set(texts)), 'duplicate new full phrases'
    semantic = [normalized(phrase) for row in training_atoms + validation_atoms + final_atoms
                for phrase in semantic_phrases(row['text'])]
    assert len(semantic) == len(set(semantic)), 'duplicate new compiler atom phrases'
    scope_sets = [
        {normalized(row['text']) for row in training_atoms},
        {normalized(row['text']) for row in validation_atoms},
        {normalized(row['text']) for row in final_atoms},
        {normalized(row['text']) for row in final_compositions},
    ]
    assert all(not (scope_sets[i] & scope_sets[j]) for i in range(len(scope_sets)) for j in range(i + 1, len(scope_sets))), 'cohort phrase leakage'
    pair_scopes = [
        {row['pair_id'] for row in training_atoms}, {row['pair_id'] for row in validation_atoms},
        {row['pair_id'] for row in final_atoms}, {row['pair_id'] for row in final_compositions},
    ]
    assert all(not (pair_scopes[i] & pair_scopes[j]) for i in range(len(pair_scopes)) for j in range(i + 1, len(pair_scopes))), 'pair-ID leakage'
    return dict(historical_and_cohort_phrase_checks='passed', new_full_texts=len(texts),
                new_semantic_phrases=len(semantic), literal_field_name_hits=0,
                historical_phrase_inclusions=0, cross_cohort_full_text_overlap=0,
                cross_cohort_pair_id_overlap=0)


def _canonical_json(value):
    return canonical(value) + b'\n'


def prepare(protocol_path=PROTOCOL):
    protocol_path = Path(protocol_path).resolve()
    cfg = load(protocol_path)
    assert cfg['field_definitions'] == EXPECTED_DEFINITIONS, 'field definitions changed from the frozen CBF8 protocol'
    assert tuple(cfg['schema']) == tuple(EXPECTED_DEFINITIONS)
    source_info = _check_c7_source(cfg)
    c6_root, c6_inventory = _source_copy_and_reference_check(cfg, source_info)
    references, reference_lineage = _references(cfg, c6_root, c6_inventory)

    source = Path(cfg['source_root'])
    root = Path(cfg['output_root'])
    assert not root.exists(), f'Refusing existing CBF8 evidence root: {root}'
    # Both corpus builders are deterministic, pure, in-memory generators.
    from schema_support_training_corpus import build_training
    from schema_support_final_corpus import build_final
    semantic_training, semantic_validation, training_audit = build_training()
    final_atoms, final_compositions, final_audit = build_final()

    literal_training = load(source / 'training_atoms.json')
    literal_validation = load(source / 'validation_atoms.json')
    g0 = load(source / 'literal_holdout_atoms.json')
    development = load(source / 'development_atoms.json')
    development_compositions = load(source / 'development_compositions.json')
    literals = load(source / 'literal_cases.json')
    states = load(source / 'states.json')
    assert len(literal_training) == len(literal_validation) == len(g0) == len(development) == 32
    assert len(development_compositions) == 50 and len(literals) == 155 and len(states) == 64
    assert sum(state['source_split'] != 'train' for state in states) == 32
    assert len(semantic_training) == 64 and len(semantic_validation) == 32
    train_pairs = _check_groups(semantic_training, 64, 32, 8, 'semantic training')
    validation_pairs = _check_groups(semantic_validation, 32, 16, 4, 'semantic validation')
    final_atom_pairs = _check_groups(final_atoms, 128, 64, 16, 'fresh final atoms')
    final_comp_pairs = defaultdict(list)
    for row in final_compositions:
        final_comp_pairs[row['pair_id']].append(row)
    assert len(final_compositions) == 128 and len(final_comp_pairs) == 64
    assert all(len(group) == 2 and group[0]['weights'] == [-w for w in group[1]['weights']]
               for group in final_comp_pairs.values()), 'final composition reversal pairs'
    assert not (set(train_pairs) & set(validation_pairs) or set(train_pairs) & set(final_atom_pairs) or
                set(validation_pairs) & set(final_atom_pairs) or set(final_comp_pairs) & set(final_atom_pairs)), 'pair scope overlap'

    compiler_rows, atom_uses, coverage = _compile_checks(
        semantic_training, semantic_validation, final_atoms, final_compositions)
    lexical = _lexical_checks(semantic_training + semantic_validation + final_atoms + final_compositions,
                              semantic_training, semantic_validation, final_atoms, final_compositions, references)
    assert training_audit['validation_usage'].lower().find('diagnostic') >= 0
    assert training_audit['template_separation']['disjoint'] is True
    assert final_audit['independent_author_provenance']['author']
    for audit, keys in (
            (training_audit['lexical_checks'], ('normalized_duplicates', 'normalized_atomic_semantic_duplicates',
                                                'literal_schema_name_hits', 'complete_old_phrase_inclusions')),
            (final_audit['lexical_checks'], ('normalized_full_text_duplicates',
                                              'normalized_atomic_semantic_duplicates',
                                              'literal_schema_name_hits', 'complete_historical_phrase_inclusions',
                                              'exact_C8_training_validation_phrase_collisions'))):
        assert all(not audit[key] for key in keys), f'generator lexical audit failed: {keys}'

    definitions_json = json.dumps(cfg['field_definitions'], sort_keys=True, indent=2, ensure_ascii=False) + '\n'
    definitions_sha = sha_bytes(definitions_json.encode('utf-8'))
    assert training_audit['provenance']['field_definitions_sha256'] == definitions_sha
    assert final_audit['protocol']['sha256'] == sha(protocol_path)
    training_audit = dict(training_audit, parent_preparation_checks=dict(
        generated_training_atoms=64, generated_validation_atoms=32,
        reversal_groups_training=len(train_pairs), reversal_groups_validation=len(validation_pairs),
        compiler_ast_weight_checks=[r for r in compiler_rows if r['cohort'] in ('semantic_training', 'semantic_validation')],
        lexical=lexical, fixed_field_definitions_sha256=definitions_sha,
        semantic_validation_never_selects_checkpoint_or_arm=True))
    final_audit = dict(final_audit, parent_preparation_checks=dict(
        compiler_ast_weight_checks=[r for r in compiler_rows if r['cohort'] in ('final_atoms', 'final_compositions')],
        atom_composition_uses=atom_uses, composition_coverage=coverage, lexical=lexical,
        fixed_field_definitions_sha256=definitions_sha,
        independent_meaning_audit_not_created=True))

    root.mkdir(parents=True)
    copied = {}
    for name in COPY_FILES:
        payload = (source / name).read_bytes()
        put(root, name, payload)
        copied[name] = sha_bytes(payload)
        assert payload == (source / name).read_bytes()
    generated = {
        'semantic_training_atoms.json': semantic_training,
        'semantic_validation_atoms.json': semantic_validation,
        'final_atoms.json': final_atoms,
        'final_compositions.json': final_compositions,
        'training_corpus_audit.json': training_audit,
        'final_corpus_audit.json': final_audit,
        'compiler_checks.json': compiler_rows,
    }
    for name, value in generated.items():
        put(root, name, _canonical_json(value))

    # Only opaque random IDs and text are exposed to judges; expected labels remain sealed.
    review_rows = (
        [('semantic_training', row) for row in semantic_training] +
        [('semantic_validation', row) for row in semantic_validation] +
        [('final_atoms', row) for row in final_atoms] +
        [('final_compositions', row) for row in final_compositions]
    )
    final_by_id = {row['id']: row for row in final_atoms}
    questions, key = [], {}
    for cohort, row in review_rows:
        review_id = uuid.uuid4().hex
        questions.append(dict(review_id=review_id, text=row['text']))
        expected = dict(weights=row['weights'])
        if 'axis' in row:
            expected.update(axis=row['axis'], sign=row['sign'])
        if 'components' in row:
            expected['components'] = [
                dict(axis=final_by_id[component['atom_id']]['axis'],
                     sign=final_by_id[component['atom_id']]['sign'],
                     factor=component['factor'])
                for component in row['components']
            ]
        key[review_id] = dict(cohort=cohort, record_id=row['id'], expected=expected)
    import random
    random.SystemRandom().shuffle(questions)
    packet = dict(
        schema_version='cbf8-blind-meaning-review-v1',
        instructions=[
            'Judge each question independently using only its text and the fixed field definitions below.',
            'Do not infer or request its cohort, source record ID, expected answer, rationale, or source file.',
            'Return decision=accept only when the meaning is clear and complete; otherwise use ambiguous or reject.',
            'For an atomic criterion, identify exactly one canonical field and sign: +1 means more of that field is preferred; -1 means less is preferred.',
            'For a two-term composition, report each term left-to-right with its canonical field axis and sign, then report the exact four-field integer weight vector in axis_order.',
            'The four-vector is the signed weighted sum of the two terms, with zero for unmentioned fields.',
            'Exact syntax, not semantic inference, owns numeric factors. Under the frozen criterion grammar, an unweighted A; also B join defines coefficient one for each term. Quantities inside either preference are context, not extra weights. Explicit weight-one/weight-two joins use their stated coefficients.',
            'Classify the whole question: every two-term join is a composition, even if either term alone is a clear atomic preference. Judge each term axis/sign independently; flag genuine meaning ambiguity rather than guessing. Numeric-factor uncertainty under the stated exact grammar is not semantic ambiguity.',
        ],
        axis_order=list(cfg['schema']), field_definitions=cfg['field_definitions'],
        response_schema=dict(
            common=dict(review_id='copy the opaque review_id exactly',
                        decision=['accept', 'ambiguous', 'reject'],
                        kind=['atomic', 'composition']),
            atomic=dict(axis='zero-based index into axis_order', sign=[-1, 1]),
            composition=dict(
                components=dict(count=2, order='left-to-right in surface text',
                                item=dict(axis='zero-based index into axis_order', sign=[-1, 1])),
                weights=dict(count=4, type='integer', order='axis_order')),
        ),
        questions=questions,
    )
    ids = sorted(row['review_id'] for row in questions)
    assert len(ids) == len(set(ids)) == 352
    put(root, 'blind_meaning_questions.json', _canonical_json(packet))
    put(root, 'blind_meaning_review_ids.json', _canonical_json(ids))
    put(root, 'blind_meaning_key.json', _canonical_json(key))
    assert not (root / 'meaning_audit.json').exists(), 'Preparation MUST NOT fabricate an independent meaning audit'

    baseline_path = root / 'baseline_references.json'
    put(root, baseline_path.name, _canonical_json(source_info['baseline_references']))
    packet_sha = sha(root / 'blind_meaning_questions.json')
    ids_sha = sha(root / 'blind_meaning_review_ids.json')
    key_sha = sha(root / 'blind_meaning_key.json')
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE.parents[1], text=True).strip()
    source_hashes = dict(source_info['source_verified_files'])
    source_hashes.update({f'lexical_reference:{k}': v['sha256'] for k, v in reference_lineage.items()})
    artifacts = list(COPY_FILES) + list(generated) + [
        'blind_meaning_questions.json', 'blind_meaning_review_ids.json',
        'blind_meaning_key.json', 'baseline_references.json',
    ]
    manifest = dict(
        experiment='CBF8', protocol_sha256=sha(protocol_path), protocol_path=str(protocol_path),
        output_root=str(root), source_root=str(source), source_result_manifest_sha256=source_info['source_result_manifest_sha256'],
        source_results_sha256=source_info['source_results_sha256'],
        source_verification_sha256=source_info['source_verification_sha256'],
        source_corpus_manifest_sha256=source_info['source_corpus_manifest_sha256'],
        source_measurement_git_revision=source_info['source_measurement_git_revision'],
        source_inherited_measurement_git_revision=source_info['source_inherited_measurement_git_revision'],
        preparation_git_revision=git_revision,
        preparation_source_sha256={name: sha(HERE / name) for name in (
            'schema_support_prepare.py', 'schema_support_encoder.py', 'schema_support_run.py',
            'schema_support_training_corpus.py', 'schema_support_final_corpus.py',
            'schema_support_meaning.py',
            'schema_support_protocol.json', 'schema_grounding_compiler.py', 'schema_relation_evaluate.py',
            'relation_pretrained_run.py', 'relation_pretrained_encoder.py', 'audit.py', 'build.py')},
        source_verified_files=source_hashes,
        counts=dict(literal_training=32, literal_validation=32, literal_holdout=32,
                    semantic_training=64, semantic_validation=32, development_atoms=32,
                    development_compositions=50, literal_cases=155, states=64,
                    final_atoms=128, final_compositions=128, final_atom_reversal_pairs=64,
                    final_composition_reversal_pairs=64, blind_review_rows=len(review_rows)),
        blind_review=dict(
            schema_version=packet['schema_version'], question_file='blind_meaning_questions.json',
            packet_sha256=packet_sha, review_ids_file='blind_meaning_review_ids.json',
            review_ids_sha256=ids_sha, key_file='blind_meaning_key.json',
            key_sha256=key_sha, predictions_file='blind_meaning_predictions.json',
            expected_questions=len(review_rows)),
        fixed_field_definitions=cfg['field_definitions'], field_definitions_sha256=definitions_sha,
        baseline_references=source_info['baseline_references'], final_closed_until_selection=True,
        meaning_audit=dict(required=True, supplied_by='independent blind auditor',
                           created_by_preparation=False, before_model_execution=True,
                           review_cohorts=['semantic_training', 'semantic_validation',
                                           'final_atoms', 'final_compositions']),
        lexical_reference_lineage=reference_lineage,
        files={name: sha(root / name) for name in artifacts},
    )
    put(root, 'corpus_manifest.json', _canonical_json(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(prepare(args.protocol), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
