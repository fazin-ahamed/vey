"""CBF-8 shipping-train semantic atoms and diagnostic validation corpus.

This module only authors deterministic in-memory rows and lexical provenance. It
never writes cohort artifacts or reads model outputs.
"""
import json
from collections import Counter
from pathlib import Path

from build import canonical
from relation_pretrained_corpus import _references as _prior_references
from schema_grounding_compiler import AXES, parse_criterion
from schema_relation_corpus import (
    _STOP,
    _contains,
    _json_bytes,
    _normalized,
    _overlap,
    _semantic_phrases,
    _sha,
    _tokens,
)

PROTOCOL = Path(__file__).with_name('schema_support_protocol.json')
RELATION_PROTOCOL = Path(__file__).with_name('relation_pretrained_protocol.json')

# Each tuple is (template_id, template, positive direction, negative direction,
# quantity, unit, fixed exposure, rationale). The only text substitution within
# a reversal pair is the direction word. Training and validation templates are
# separately authored and use different quantities/exposure descriptions.
TRAINING_PAIRS = (
    (
        (
            ('rel-tr-01', 'Prefer {direction} functional failures per 1000 matched-use cycles.', 'fewer', 'more',
             'functional failures', 'events per 1000 cycles',
             'The same 1000 cycles of the same intended task are observed for each item.',
             'A smaller failure frequency under equal task exposure means more successful functioning.'),
            ('rel-tr-02', 'Choose the {direction} share of 120 ordinary sessions ended by an internal shutdown before the assigned task could finish.', 'smaller', 'larger',
             'sessions ended by a task-stopping internal shutdown', 'share of sessions',
             'Each item is observed for the same 120 ordinary-use sessions of its assigned task.',
             'The shutdown prevents completion of the assigned task; fewer such failures at fixed exposure mean greater reliability.'),
            ('rel-tr-03', 'Prefer {direction} no-start events in 250 power-on attempts.', 'fewer', 'more',
             'no-start events', 'events per 250 attempts',
             'Both items receive the same 250 power-on attempts under matched conditions.',
             'A no-start event is failure to begin the intended function; fewer events per fixed attempts means greater reliability.'),
            ('rel-tr-04', 'Choose a {direction} rate of mechanism faults that stop the intended task per 800 operating hours.', 'lower', 'higher',
             'task-stopping mechanism faults', 'faults per operating hour',
             'The comparison uses the same 800 operating hours at the same task and load.',
             'Faults in the mechanism are functional failures; a lower rate for equal operating exposure means greater reliability.'),
            ('rel-tr-05', 'Prefer {direction} runs stopped by a component failure out of 300 matched runs.', 'fewer', 'more',
             'runs stopped by a component failure', 'runs per 300 matched runs',
             'Each item performs the same 300 matched runs of the intended task.',
             'A component failure stopping a run is loss of function; fewer stopped runs means greater reliability.'),
            ('rel-tr-06', 'Choose a {direction} percentage of 400 trials interrupted by loss of function.', 'smaller', 'larger',
             'trials interrupted by loss of function', 'percentage of trials',
             'The denominator is the same 400 trials under matched operating conditions.',
             'Interruption by loss of function is a failure outcome; a smaller fixed-denominator percentage means greater reliability.'),
            ('rel-tr-07', 'Prefer {direction} breakdown incidents in 90 sessions of the same task.', 'fewer', 'more',
             'breakdown incidents', 'incidents per 90 sessions',
             'The same 90 sessions of the same task are counted for both items.',
             'A breakdown is a functional failure during a session; fewer incidents at equal exposure means greater reliability.'),
            ('rel-tr-08', 'Choose a {direction} proportion of 60 work shifts ending without an equipment fault.', 'greater', 'smaller',
             'work shifts completed without an equipment fault', 'proportion of shifts',
             'Each item is observed for the same 60 matched work shifts.',
             'Completing a shift without equipment failure is successful functioning; a greater success proportion means greater reliability.'),
        ),
        (
            ('rel-va-01', 'Prefer a {direction} probability that the unit completes a fixed 12-hour endurance run before its first functional stop.', 'higher', 'lower',
             'completion before first functional stop', 'probability per endurance run',
             'The run lasts 12 hours under the same specified task and load for both items.',
             'Completion before a functional stop measures sustained function over equal exposure; higher probability means greater reliability.'),
            ('rel-va-02', 'Choose a {direction} percentage of 200 daily-use periods without a device failure.', 'greater', 'smaller',
             'daily-use periods without a device failure', 'percentage of periods',
             'Both items are observed for the same 200 daily-use periods with the same task schedule.',
             'Failure-free periods indicate successful functioning; a greater share under equal exposure means greater reliability.'),
            ('rel-va-03', 'Prefer {direction} fault-triggered reset events per 75 controlled task sequences.', 'fewer', 'more',
             'fault-triggered reset events', 'events per 75 sequences',
             'Each item completes the same 75 controlled sequences with identical reset rules.',
             'A reset required by a fault is a loss-of-function event; fewer such resets means greater reliability.'),
            ('rel-va-04', 'Choose a {direction} rate of unexpected service-ending faults across 40 identical work shifts.', 'lower', 'higher',
             'unexpected service-ending faults', 'faults per 40 shifts',
             'Both items are compared over the same 40 identical shifts and task conditions.',
             'A fault ending service is a functional failure; a lower rate at fixed exposure means greater reliability.'),
        ),
    ),
    (
        (
            ('buy-tr-01', 'Prefer {direction} money paid in the initial ownership handoff.', 'more', 'less',
             'money paid in initial ownership handoff', 'currency per initial transfer',
             'Only the single initial ownership transfer is included; later use payments are excluded.',
             'The one-time transfer payment is acquisition money; a larger amount means greater purchase expense.'),
            ('buy-tr-02', 'Choose a {direction} total on the initial sales invoice.', 'greater', 'smaller',
             'initial sales invoice total', 'currency per initial invoice',
             'The invoice covers only the original sale, not any later service or use.',
             'The original sale invoice measures the initial acquisition payment; a greater total means greater purchase expense.'),
            ('buy-tr-03', 'Prefer a {direction} single charge to obtain ownership.', 'higher', 'lower',
             'single charge to obtain ownership', 'currency per acquisition charge',
             'The charge is assessed once when ownership is obtained.',
             'A one-time charge for obtaining ownership is acquisition spending; a higher amount means greater purchase expense.'),
            ('buy-tr-04', 'Choose a {direction} amount due at the first checkout for the item.', 'larger', 'smaller',
             'amount due at first checkout', 'currency per checkout',
             'The same item is acquired in one checkout and no later payment is counted.',
             'The checkout amount pays for the initial transaction; a larger amount means greater purchase expense.'),
            ('buy-tr-05', 'Prefer {direction} cash exchanged for a single acquisition.', 'more', 'less',
             'cash exchanged for one acquisition', 'currency per transaction',
             'Each comparison concerns one initial acquisition transaction only.',
             'Cash exchanged for acquisition is an initial payment; more money means greater purchase expense.'),
            ('buy-tr-06', 'Choose a {direction} complete seller charge for the initial ownership handover.', 'higher', 'lower',
             'complete seller charge for initial ownership', 'currency per initial ownership transaction',
             'The charge is the complete payment in the initial transaction transferring ownership.',
             'The complete charge acquires the item at the initial transfer; a higher charge means greater purchase expense.'),
            ('buy-tr-07', 'Prefer a {direction} amount billed for the original transfer of ownership.', 'greater', 'smaller',
             'amount billed for original ownership transfer', 'currency per transfer',
             'The bill records the original transfer only and excludes later use or maintenance payments.',
             'The amount billed for initial ownership transfer is acquisition spending; greater billing means greater purchase expense.'),
            ('buy-tr-08', 'Choose a {direction} one-time payment required to acquire the item.', 'larger', 'smaller',
             'one-time payment to acquire the item', 'currency per acquisition',
             'The payment occurs once at acquisition and is not repeated during use.',
             'A one-time acquisition payment is initial spending; a larger payment means greater purchase expense.'),
        ),
        (
            ('buy-va-01', 'Prefer a {direction} initial amount on the order receipt before any use.', 'higher', 'lower',
             'initial amount on order receipt', 'currency per initial order',
             'The receipt covers the initial order before the item is used.',
             'The initial order amount is paid to acquire the item; a higher amount means greater purchase expense.'),
            ('buy-va-02', 'Choose a {direction} single debit when ownership transfers to the buyer.', 'larger', 'smaller',
             'single debit at transfer of property', 'currency per ownership transfer',
             'Only the debit at the initial transfer is measured.',
             'The single debit is money paid to acquire ownership; a larger debit means greater purchase expense.'),
            ('buy-va-03', 'Prefer {direction} dollars required to complete the first sale.', 'more', 'less',
             'dollars required to complete first sale', 'dollars per initial sale',
             'The amount is fixed to the first sale and excludes subsequent operating payments.',
             'Dollars required for the first sale are acquisition money; more dollars mean greater purchase expense.'),
            ('buy-va-04', 'Choose a {direction} one-off transaction total for receiving ownership.', 'greater', 'smaller',
             'one-off transaction total for ownership', 'currency per transaction',
             'The total concerns one initial ownership transaction under the same included-fee rules.',
             'A one-off total paid to receive ownership is acquisition spending; a greater total means greater purchase expense.'),
        ),
    ),
    (
        (
            ('oper-tr-01', 'Prefer a {direction} cash fee per operating hour.', 'higher', 'lower',
             'cash fee per operating hour', 'currency per hour',
             'The rate is compared over the same operating hours and task conditions.',
             'A monetary fee charged during use is ongoing spending; a higher hourly fee means greater operating expense.'),
            ('oper-tr-02', 'Choose a {direction} recurring bill for each active month.', 'larger', 'smaller',
             'recurring bill per active month', 'currency per month',
             'The same number of active months is compared and the bill recurs during use.',
             'A recurring bill during active use is ongoing spending; a larger bill means greater operating expense.'),
            ('oper-tr-03', 'Prefer {direction} money for every 100 operating cycles.', 'more', 'less',
             'money for 100 operating cycles', 'currency per 100 cycles',
             'The comparison covers the same 100 completed operating cycles.',
             'Money charged for operation recurs with cycles; more money per fixed cycles means greater operating expense.'),
            ('oper-tr-04', 'Choose a {direction} total charge for 50 hours of active use.', 'higher', 'lower',
             'total charge for active use', 'currency per 50 hours',
             'Each item is used for the same 50 hours under the same task conditions.',
             'The charge is incurred during use; a higher total for equal hours means greater operating expense.'),
            ('oper-tr-05', 'Prefer a {direction} fee for each session the item is used.', 'larger', 'smaller',
             'fee per use session', 'currency per session',
             'The same number of sessions is compared and the fee applies on each use.',
             'A fee repeated per use session is ongoing spending; a larger fee means greater operating expense.'),
            ('oper-tr-06', 'Choose {direction} money per week of continued operation.', 'more', 'less',
             'money per week of operation', 'currency per operating week',
             'The same active operating weeks are compared.',
             'Money paid weekly to keep operating is ongoing spending; more money means greater operating expense.'),
            ('oper-tr-07', 'Prefer a {direction} service payment for every active month of use.', 'larger', 'smaller',
             'monthly service payment during use', 'currency per month',
             'The payment recurs monthly throughout the same use period.',
             'The service payment is monetary and recurs during use; a larger payment means greater operating expense.'),
            ('oper-tr-08', 'Choose a {direction} total monetary charge to continue using the item throughout a fixed 600-hour period.', 'higher', 'lower',
             'total continued-use charge over 600 hours', 'currency per 600 operating hours',
             'The same 600 hours of continued use are compared; the initial acquisition payment is excluded.',
             'The total charge is money incurred after acquisition for equal continued use; a higher charge means greater operating expense.'),
        ),
        (
            ('oper-va-01', 'Prefer {direction} money billed for each fixed 24-hour operating period.', 'more', 'less',
             'money billed per operating period', 'currency per 24 operating hours',
             'Both items are measured over the same 24 hours of operation.',
             'A bill incurred during operation is ongoing spending; more money for equal operating time means greater operating expense.'),
            ('oper-va-02', 'Choose a {direction} recurring amount for every 100 completed uses.', 'greater', 'smaller',
             'recurring amount per 100 uses', 'currency per 100 uses',
             'The same 100 uses are completed under matched conditions and charges recur with use.',
             'A recurring monetary amount per use is ongoing spending; a greater amount means greater operating expense.'),
            ('oper-va-03', 'Prefer a {direction} annual charge during 12 months of continued use.', 'larger', 'smaller',
             'annual charge during continued use', 'currency per year',
             'The charge covers the same 12-month period of continued use.',
             'A charge paid to continue use is ongoing spending; a larger annual charge means greater operating expense.'),
            ('oper-va-04', 'Choose a {direction} monetary fee per fixed 500 task cycles.', 'higher', 'lower',
             'monetary fee per task cycle', 'currency per 500 cycles',
             'The same 500 task cycles are completed for each item.',
             'A monetary fee incurred during repeated use is ongoing spending; a higher fee at equal exposure means greater operating expense.'),
        ),
    ),
    (
        (
            ('conv-tr-01', 'Prefer {direction} hand-operated steps to finish one standard task.', 'fewer', 'more',
             'hand-operated steps', 'steps per task',
             'The same standard task is completed once by the user for each item.',
             'Required hand-operated steps measure user work; fewer steps mean less effort and greater convenience.'),
            ('conv-tr-02', 'Choose {direction} setup actions before each use.', 'fewer', 'more',
             'user setup actions', 'actions per use',
             'Setup is measured before the same task on each use.',
             'Required setup actions are user work; fewer actions mean greater convenience.'),
            ('conv-tr-03', 'Prefer {direction} active monitoring minutes during a 30-minute run.', 'fewer', 'more',
             'active monitoring time', 'minutes per 30-minute run',
             'The same 30-minute task run is observed, counting active attention rather than passive waiting.',
             'Active monitoring is user attention effort; less active attention under the same run conditions means greater convenience.'),
            ('conv-tr-04', 'Choose {direction} control presses needed for one complete job.', 'fewer', 'more',
             'control presses', 'presses per job',
             'The same complete job is performed under the same control setup.',
             'Required control presses are user actions; fewer presses mean less effort and greater convenience.'),
            ('conv-tr-05', 'Prefer {direction} joules of physical work exerted by the user per 10 routine operations.', 'fewer', 'more',
             'physical work exerted by user', 'joules per 10 operations',
             'The same 10 routine operations are completed under the same operating conditions.',
             'The measured quantity is physical work exerted by the user; less work means greater convenience.'),
            ('conv-tr-06', 'Choose {direction} user decisions needed during a five-step setup.', 'fewer', 'more',
             'decisions made by user during setup', 'decisions per setup',
             'The same five-step setup task is used for each item.',
             'Required user decisions are cognitive work; fewer decisions mean less effort and greater convenience.'),
            ('conv-tr-07', 'Prefer {direction} active preparation time for each 20-minute use.', 'less', 'more',
             'active preparation time', 'minutes per 20-minute use',
             'The same 20-minute use task is prepared in each comparison; only active preparation is timed.',
             'Active preparation is user work rather than passive delay; less preparation time means greater convenience.'),
            ('conv-tr-08', 'Choose {direction} fields the operator must enter per session.', 'fewer', 'more',
             'manually entered fields', 'fields per session',
             'The same session and required record are used for each item.',
             'Manually entering fields is user interaction work; fewer required entries mean greater convenience.'),
        ),
        (
            ('conv-va-01', 'Prefer {direction} manual adjustments needed across a 20-minute automatic cycle.', 'fewer', 'more',
             'manual adjustments during automatic cycle', 'adjustments per 20-minute cycle',
             'Each item runs for the same 20-minute automatic cycle on the same task.',
             'Required manual adjustments are user actions; fewer adjustments mean greater convenience.'),
            ('conv-va-02', 'Choose {direction} instruction pages the user must consult during five familiarization uses.', 'fewer', 'more',
             'instruction pages consulted by user', 'pages per five uses',
             'The same five familiarization uses and instruction set are used in both comparisons.',
             'Consulting instructions requires user attention; fewer pages needed mean less effort and greater convenience.'),
            ('conv-va-03', 'Prefer {direction} hand force in newtons needed to complete one fixed setup procedure.', 'lower', 'higher',
             'hand force needed for setup', 'newtons per setup procedure',
             'The same fixed setup procedure is performed under the same conditions.',
             'Required hand force is physical effort by the user; less force means greater convenience.'),
            ('conv-va-04', 'Choose {direction} separate components the user must connect before a standard run.', 'fewer', 'more',
             'components manually connected by user', 'components per run setup',
             'The same standard run is prepared once under the same setup requirements.',
             'Connecting components is required user setup work; fewer components mean less effort and greater convenience.'),
        ),
    ),
)

REJECTED_AMBIGUITY_CANDIDATES = (
    ('Prefer longer intervals between service visits.',
     'Service timing can reflect maintenance policy, failure rate, or user effort and does not isolate one field.'),
    ('Choose less money spent over a year.',
     'A yearly total can combine the initial transaction with repeated use payments.'),
    ('Prefer fewer stops during a workday.',
     'Stops may be caused by device faults, task scheduling, or user breaks; the cause is unspecified.'),
    ('Choose a lighter item to carry.',
     'Mass may affect portability but does not directly define the frozen effort measure for the intended task.'),
    ('Prefer fewer repairs.',
     'Repair frequency can indicate functional failures, maintenance policy, or recurring monetary spending.'),
    ('Choose faster setup.',
     'Elapsed setup speed does not distinguish active user effort from passive waiting or machine performance.'),
    ('Choose a higher monetary tariff per kilowatt-hour.',
     'Energy-unit price is not total spending over equal use: candidates can consume different energy quantities.'),
    ('Choose a higher seller charge recorded at delivery.',
     'The charge can be a delivery-service fee rather than the complete initial ownership transaction.'),
)

CURRENT_SOURCE_FILES = {
    'training_atoms.json': 32,
    'validation_atoms.json': 32,
    'literal_holdout_atoms.json': 32,
    'states.json': 64,
    'literal_cases.json': 155,
    'final_atoms.json': 128,
    'final_compositions.json': 128,
}
SOURCE_ATOM_KEYS = ('axis', 'id', 'provenance', 'sign', 'text')


def _read_rows(path, expected_count):
    payload = path.read_bytes()
    rows = json.loads(payload)
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f'Expected {expected_count} JSON rows in {path}')
    lineage = dict(path=str(path), sha256=_sha(payload), rows=len(rows))
    return rows, lineage


def _source_lineage(source):
    rows_by_name, lineage = {}, {}
    for filename, expected_count in CURRENT_SOURCE_FILES.items():
        rows, record = _read_rows(source / filename, expected_count)
        rows_by_name[filename] = rows
        lineage[filename] = record
    for filename in ('training_atoms.json', 'validation_atoms.json', 'literal_holdout_atoms.json'):
        rows = rows_by_name[filename]
        if any(tuple(sorted(row)) != SOURCE_ATOM_KEYS for row in rows):
            raise ValueError(f'Unexpected source atom row schema in {source / filename}')
    return rows_by_name, lineage


def _references(current_rows):
    previous_protocol_bytes = RELATION_PROTOCOL.read_bytes()
    previous_protocol = json.loads(previous_protocol_bytes)
    previous_source = Path(previous_protocol['source_root'])
    groups, previous_lineage = _prior_references(previous_source)

    # CBF-7's final atoms and compositions are references for lexical exclusion
    # only. No CBF-7 model outputs, scores, or selection artifacts are read.
    current_final = [
        dict(id=row['id'], text=row['text'])
        for filename in ('final_atoms.json', 'final_compositions.json')
        for row in current_rows[filename]
    ]
    groups['old_CBF7_final'] = sorted(current_final, key=lambda row: row['id'])
    source_lineage = {
        'relation_pretrained_protocol': dict(
            path=str(RELATION_PROTOCOL), sha256=_sha(previous_protocol_bytes), rows=None),
        'previous_reference_files': previous_lineage,
    }
    for values in groups.values():
        values.sort(key=lambda row: row['id'])
    return groups, source_lineage


def _pair_metadata(scope, axis, variant, pair):
    template_id, template, positive, negative, quantity, unit, exposure, rationale = pair
    if template.count('{direction}') != 1:
        raise ValueError(f'Template must contain one direction slot: {template_id}')
    positive_text = template.format(direction=positive)
    negative_text = template.format(direction=negative)
    pair_id = f'c8-{scope}-pair-{axis}-{variant:02d}'
    prefix = 'c8-train-atom' if scope == 'train' else 'c8-validation-atom'
    positive_id = f'{prefix}-{axis}-higher-{variant:02d}'
    negative_id = f'{prefix}-{axis}-lower-{variant:02d}'
    return dict(
        pair_id=pair_id,
        template_id=template_id,
        template=template,
        template_signature=_normalized(template.replace('{direction}', 'direction')),
        quantity=quantity,
        unit=unit,
        fixed_exposure=exposure,
        positive_atom_id=positive_id,
        negative_atom_id=negative_id,
        positive_text=positive_text,
        negative_text=negative_text,
        rationale=rationale,
        reversal='Only the comparative direction changes; the measured quantity, unit, task, and exposure remain fixed.',
    )


def _build_atoms(scope, pairs_by_axis):
    atoms, meanings = [], []
    stratum = 'semantic_train' if scope == 'train' else 'semantic_validation'
    for axis, pair_group in enumerate(pairs_by_axis):
        for variant, pair in enumerate(pair_group):
            meaning = _pair_metadata(scope, axis, variant, pair)
            for sign, atom_id, text in (
                (1, meaning['positive_atom_id'], meaning['positive_text']),
                (-1, meaning['negative_atom_id'], meaning['negative_text']),
            ):
                weights = [0] * len(AXES)
                weights[axis] = sign
                parsed = parse_criterion(text)
                if len(parsed) != 1 or parsed[0]['factor'] != 1 or parsed[0]['literal_axis'] is not None:
                    raise ValueError(f'Expected one nonliteral unit atom: {atom_id}')
                if any(_contains(_tokens(text), _tokens(field)) for field in AXES):
                    raise ValueError(f'Literal schema field name in semantic atom: {atom_id}')
                atoms.append(dict(id=atom_id, text=text, axis=axis, sign=sign,
                                  weights=weights, pair_id=meaning['pair_id'],
                                  variant=variant, stratum=stratum))
            meanings.append(dict(axis=axis, canonical_field=AXES[axis], **meaning))
    return atoms, meanings


def _lexical_audit(atoms, references):
    normalized_duplicates = [
        dict(normalized=text, count=count)
        for text, count in sorted(Counter(_normalized(row['text']) for row in atoms).items())
        if count > 1
    ]
    semantic_duplicates = [
        dict(normalized=text, count=count)
        for text, count in sorted(Counter(
            _normalized(_semantic_phrases(row['text'])[0]) for row in atoms
        ).items())
        if count > 1
    ]
    byte_exact_matches, normalized_reference_matches = [], []
    complete_old_phrase_inclusions = []
    reference_phrases = []
    for group, rows in references.items():
        for row in rows:
            reference_phrases.append((group, row['id'], 'full_text', row['text']))
            for phrase in _semantic_phrases(row['text']):
                reference_phrases.append((group, row['id'], 'semantic_phrase', phrase))

    for atom in atoms:
        atom_tokens = _tokens(atom['text'])
        atom_normalized = _normalized(atom['text'])
        for group, rows in references.items():
            for row in rows:
                if atom['text'] == row['text']:
                    byte_exact_matches.append(dict(atom_id=atom['id'], reference_group=group,
                                                    source_id=row['id']))
                if atom_normalized == _normalized(row['text']):
                    normalized_reference_matches.append(dict(atom_id=atom['id'],
                                                             reference_group=group,
                                                             source_id=row['id'],
                                                             phrase_kind='full_text'))
        for group, source_id, phrase_kind, phrase in reference_phrases:
            phrase_tokens = _tokens(phrase)
            if atom_tokens == phrase_tokens:
                normalized_reference_matches.append(dict(atom_id=atom['id'],
                                                         reference_group=group,
                                                         source_id=source_id,
                                                         phrase_kind=phrase_kind))
            if _contains(atom_tokens, phrase_tokens):
                complete_old_phrase_inclusions.append(dict(
                    atom_id=atom['id'], reference_group=group, source_id=source_id,
                    phrase_kind=phrase_kind, phrase=phrase))

    literal_schema_name_hits = [
        dict(id=atom['id'], field=field)
        for atom in atoms for field in AXES
        if _contains(_tokens(atom['text']), _tokens(field))
    ]
    token_overlap = [
        dict(id=atom['id'], stratum=atom['stratum'], reference_group=group,
             **_overlap(atom['text'], rows))
        for atom in atoms for group, rows in references.items()
    ]
    return dict(
        normalized_duplicates=normalized_duplicates,
        normalized_atomic_semantic_duplicates=semantic_duplicates,
        literal_schema_name_hits=literal_schema_name_hits,
        byte_exact_reference_matches=byte_exact_matches,
        normalized_reference_phrase_matches=normalized_reference_matches,
        complete_old_phrase_inclusions=complete_old_phrase_inclusions,
        token_overlap=token_overlap,
        overlap_definition=dict(
            normalization='Casefold; ASCII alphanumeric tokens; punctuation and whitespace ignored.',
            content_words='Unique normalized tokens minus the recorded CBF-6 stopword set; no stemming.',
            stopwords=sorted(_STOP),
            ngrams='Contiguous normalized all-word token ngrams of lengths two through five; maximum distinct shared-ngram count per reference question.',
            ties='First source ID in lexical order.',
            policy='Descriptive only for token-overlap magnitudes; exact and complete-phrase collisions are exclusions.'),
    )


def build_training():
    """Return 64 training atoms, 32 diagnostic validation atoms, and an audit.

    The function is deterministic and read-only. Semantic validation rows are
    recorded solely for diagnostics and must never select optimizers, checkpoints,
    or arms.
    """
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    source = Path(protocol['source_root'])
    current_rows, current_lineage = _source_lineage(source)
    references, prior_lineage = _references(current_rows)
    source_cohort_identity = {}
    for filename in ('training_atoms.json', 'validation_atoms.json', 'literal_holdout_atoms.json'):
        predecessor = prior_lineage['previous_reference_files'][filename]
        identical = current_lineage[filename]['sha256'] == predecessor['sha256']
        if not identical:
            raise ValueError(f'Literal source cohort is not byte-identical to its predecessor: {filename}')
        source_cohort_identity[filename] = dict(
            byte_identical_to_predecessor=True,
            sha256=current_lineage[filename]['sha256'],
            rows=current_lineage[filename]['rows'])

    if tuple(protocol['schema']) != AXES:
        raise ValueError('CBF-8 schema order differs from the frozen compiler')
    if set(protocol['field_definitions']) != set(AXES):
        raise ValueError('CBF-8 must freeze one definition per schema field')
    if tuple(len(train) for train, _ in TRAINING_PAIRS) != (8, 8, 8, 8):
        raise ValueError('Every field requires eight semantic training reversal pairs')
    if tuple(len(validation) for _, validation in TRAINING_PAIRS) != (4, 4, 4, 4):
        raise ValueError('Every field requires four semantic validation reversal pairs')

    training_atoms, training_meanings = _build_atoms(
        'train', tuple(train for train, _ in TRAINING_PAIRS))
    validation_atoms, validation_meanings = _build_atoms(
        'validation', tuple(validation for _, validation in TRAINING_PAIRS))
    all_atoms = training_atoms + validation_atoms
    if len(training_atoms) != 64 or len(validation_atoms) != 32:
        raise ValueError('CBF-8 atom row counts differ from the frozen protocol')
    if len({atom['id'] for atom in all_atoms}) != len(all_atoms):
        raise ValueError('CBF-8 atom IDs must be unique')

    expected_counts = {
        'semantic_train': {(axis, sign): 8 for axis in range(4) for sign in (-1, 1)},
        'semantic_validation': {(axis, sign): 4 for axis in range(4) for sign in (-1, 1)},
    }
    direction_balance = {}
    for scope, atoms in (('semantic_train', training_atoms),
                         ('semantic_validation', validation_atoms)):
        observed = Counter((atom['axis'], atom['sign']) for atom in atoms)
        expected = Counter(expected_counts[scope])
        if observed != expected:
            raise ValueError(f'{scope} field/sign balance failed')
        direction_balance[scope] = [
            dict(axis=axis, canonical_field=AXES[axis], sign=sign,
                 count=observed[(axis, sign)])
            for axis in range(len(AXES)) for sign in (-1, 1)
        ]

    training_templates = {row['template_id'] for row in training_meanings}
    validation_templates = {row['template_id'] for row in validation_meanings}
    training_signatures = {row['template_signature'] for row in training_meanings}
    validation_signatures = {row['template_signature'] for row in validation_meanings}
    training_quantities = {row['quantity'] for row in training_meanings}
    validation_quantities = {row['quantity'] for row in validation_meanings}
    if (len(training_templates) != len(training_meanings) or
            len(validation_templates) != len(validation_meanings) or
            len(training_signatures) != len(training_meanings) or
            len(validation_signatures) != len(validation_meanings)):
        raise ValueError('Every semantic reversal pair requires a unique phrasing template')
    if (training_templates & validation_templates or
            training_signatures & validation_signatures):
        raise ValueError('Training and validation phrasing templates must be disjoint')
    if (len(training_quantities) != len(training_meanings) or
            len(validation_quantities) != len(validation_meanings)):
        raise ValueError('Every semantic reversal pair requires a unique measured quantity')
    if training_quantities & validation_quantities:
        raise ValueError('Training and validation measured quantities must be disjoint')

    lexical = _lexical_audit(all_atoms, references)
    if any(lexical[key] for key in (
            'normalized_duplicates', 'normalized_atomic_semantic_duplicates',
            'literal_schema_name_hits', 'byte_exact_reference_matches',
            'normalized_reference_phrase_matches', 'complete_old_phrase_inclusions')):
        raise ValueError('CBF-8 lexical exclusion failed: ' + repr({
            key: lexical[key] for key in (
                'normalized_duplicates', 'normalized_atomic_semantic_duplicates',
                'literal_schema_name_hits', 'byte_exact_reference_matches',
                'normalized_reference_phrase_matches', 'complete_old_phrase_inclusions')
        }))

    field_definitions_sha256 = _sha(_json_bytes(protocol['field_definitions']))
    current_row_schemas = {
        filename: sorted(rows[0].keys())
        for filename, rows in current_rows.items()
        if filename in ('training_atoms.json', 'validation_atoms.json', 'literal_holdout_atoms.json')
    }
    audit = dict(
        experiment=protocol['experiment'],
        license_class='shipping-train',
        provenance=dict(
            author='CBF8 training-corpus author',
            source_root=str(source),
            generator_path=str(Path(__file__)),
            generator_sha256=_sha(Path(__file__).read_bytes()),
            protocol_path=str(PROTOCOL),
            protocol_sha256=_sha(protocol_bytes),
            field_definitions_sha256=field_definitions_sha256,
            field_definitions_sha_method='SHA-256 of sorted, indented UTF-8 JSON with a final newline.',
            relation_protocol_path=str(RELATION_PROTOCOL),
            prior_reference_lineage=prior_lineage,
            source_lineage=current_lineage,
            source_cohort_identity=source_cohort_identity,
            source_cohort_row_schema=current_row_schemas,
            source_cohorts='Read-only lineage and schema checks; original literal cohorts are never modified.',
            model_output_policy='No model outputs, scores, selections, or quality summaries are read.',
        ),
        counts=dict(
            training_atoms=len(training_atoms),
            validation_atoms=len(validation_atoms),
            training_reversal_pairs=len(training_meanings),
            validation_reversal_pairs=len(validation_meanings),
            reference_groups={group: len(rows) for group, rows in references.items()},
            reference_byte_hashes={
                **{f'cbf7_source/{name}': record['sha256'] for name, record in current_lineage.items()},
                **{f'cbf6_source/{name}': record['sha256'] for name, record in prior_lineage['previous_reference_files'].items()},
            },
            reference_row_counts={
                **{f'cbf7_source/{name}': record['rows'] for name, record in current_lineage.items()},
                **{f'cbf6_source/{name}': record['rows'] for name, record in prior_lineage['previous_reference_files'].items()},
            },
        ),
        direction_balance=direction_balance,
        template_separation=dict(
            training_template_ids=sorted(training_templates),
            validation_template_ids=sorted(validation_templates),
            training_quantity_templates=sorted(training_quantities),
            validation_quantity_templates=sorted(validation_quantities),
            disjoint=True,
            reversal_construction='One authored template per pair; direction slot is the only substitution within that pair.'),
        meaning_review=dict(
            timing='Authored before CBF-8 scores; no outcome-dependent phrase filtering.',
            rationale_policy='Pair rationales and measurement conditions are provenance-only and are not present in atom rows or model inputs.',
            training_pairs=training_meanings,
            validation_pairs=validation_meanings,
            rejected_ambiguity_candidates=[dict(text=text, reason=reason)
                                           for text, reason in REJECTED_AMBIGUITY_CANDIDATES],
        ),
        lexical_checks=dict(
            reference_scope='CBF4/5 development references and CBF6/7 final references are lexical exclusions only; no outcomes are inspected.',
            **lexical,
        ),
        validation_usage='Semantic validation is diagnostic only and MUST NOT select optimizers, checkpoints, or arms.',
        output_sha256=dict(
            training_atoms_json=_sha(canonical(training_atoms) + b'\n'),
            validation_atoms_json=_sha(canonical(validation_atoms) + b'\n'),
        ),
    )
    return training_atoms, validation_atoms, audit
