"""Pure CBF-8 final-corpus builder; no model work or artifact writes."""
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

from schema_grounding_compiler import AXES, parse_criterion
from schema_relation_corpus import _compositions as _coverage_compositions
from schema_relation_corpus import _contains, _normalized, _overlap, _tokens


PROTOCOL = Path(__file__).with_name('schema_support_protocol.json')
CBF6_REFERENCE_COUNTS = {
    'final_atoms.json': 128,
    'final_compositions.json': 128,
}
CBF7_REFERENCE_COUNTS = {
    'final_atoms.json': 128,
    'final_compositions.json': 128,
}

# Each tuple is (higher-coordinate wording, lower-coordinate wording,
# direction token in each sentence, rationale). The rationale is audit-only.
# These 64 pairs were authored for the CBF-8 final pool, independently of the
# C8 training corpus. Only each pair's comparative direction changes.
MEANING_PAIRS = (
    (
        ('Prefer a lower probability that a matched attempt fails to finish its assigned task across 120 identical attempts.', 'Prefer a higher probability that a matched attempt fails to finish its assigned task across 120 identical attempts.', 'lower', 'higher', 'Failure to finish the same intended task is measured over a fixed set of matched attempts; lower probability means more dependable function.'),
        ('Prefer a smaller fraction of 80 same-condition runs in which the unit cannot perform one assigned task.', 'Prefer a larger fraction of 80 same-condition runs in which the unit cannot perform one assigned task.', 'smaller', 'larger', 'The same task and 80-run exposure are fixed; the fraction unable to complete it is the failure measure.'),
        ('Prefer fewer failures to start the same assigned task among 64 identical start attempts.', 'Prefer more failures to start the same assigned task among 64 identical start attempts.', 'fewer', 'more', 'Failure to begin the same task is measured under 64 matched starts.'),
        ('Prefer a lower chance that the device loses its ability to finish the assigned task during one standard session across 70 matched sessions.', 'Prefer a higher chance that the device loses its ability to finish the assigned task during one standard session across 70 matched sessions.', 'lower', 'higher', 'Loss of task-completion ability is the failure event; session conditions and exposure stay fixed.'),
        ('Prefer a smaller share of 90 same-condition trials where the assigned operation stops before completion.', 'Prefer a larger share of 90 same-condition trials where the assigned operation stops before completion.', 'smaller', 'larger', 'The assigned operation and 90-trial denominator remain fixed; only its failure share changes.'),
        ('Prefer fewer uses in which the unit cannot complete the same operation across 100 repeated uses.', 'Prefer more uses in which the unit cannot complete the same operation across 100 repeated uses.', 'fewer', 'more', 'Incomplete operations are compared at equal use exposure.'),
        ('Prefer a lower probability that a functional failure prevents completion of the assigned task during each of 50 matched cycles.', 'Prefer a higher probability that a functional failure prevents completion of the assigned task during each of 50 matched cycles.', 'lower', 'higher', 'The event is a task-stopping functional failure over the same 50-cycle exposure.'),
        ('Prefer fewer occasions the item cannot perform the requested task in 40 identical sessions.', 'Prefer more occasions the item cannot perform the requested task in 40 identical sessions.', 'fewer', 'more', 'The requested task and 40-session exposure define the failure frequency.'),
        ('Prefer a smaller proportion of 75 matched attempts ending after loss of the ability to carry out the assigned task.', 'Prefer a larger proportion of 75 matched attempts ending after loss of the ability to carry out the assigned task.', 'smaller', 'larger', 'The proportion of matched attempts with lost task function is the measured quantity.'),
        ('Prefer fewer trials where the mechanism stops functioning before the same assigned task ends across 55 repeated trials.', 'Prefer more trials where the mechanism stops functioning before the same assigned task ends across 55 repeated trials.', 'fewer', 'more', 'Functional stoppage before task completion is counted over the same 55 trials.'),
        ('Prefer a lower incidence of start failures for one fixed task among 48 equal start tests.', 'Prefer a higher incidence of start failures for one fixed task among 48 equal start tests.', 'lower', 'higher', 'Failure to begin the fixed task is evaluated against the same 48-start denominator.'),
        ('Prefer a lower rate of failed task completions per 200 identical requests.', 'Prefer a higher rate of failed task completions per 200 identical requests.', 'lower', 'higher', 'The requested task and exposure denominator stay fixed; the rate measures incomplete function.'),
        ('Prefer fewer runs where the item becomes unable to complete its one assigned task among 36 repeated runs.', 'Prefer more runs where the item becomes unable to complete its one assigned task among 36 repeated runs.', 'fewer', 'more', 'The task remains fixed over 36 repetitions; inability to finish it is a failure.'),
        ('Prefer a lower probability that a functional fault stops the assigned operation before completion during each of 100 same-exposure trials.', 'Prefer a higher probability that a functional fault stops the assigned operation before completion during each of 100 same-exposure trials.', 'lower', 'higher', 'Only a task-stopping fault is counted; trial exposure is matched.'),
        ('Prefer fewer attempts that end because the item cannot continue performing the assigned task among 72 trials.', 'Prefer more attempts that end because the item cannot continue performing the assigned task among 72 trials.', 'fewer', 'more', 'The task and 72-trial denominator isolate loss of function during execution.'),
        ('Prefer a smaller percentage of 50 matched sessions where the unit does not complete its intended task.', 'Prefer a larger percentage of 50 matched sessions where the unit does not complete its intended task.', 'smaller', 'larger', 'The intended task and session count are constant; the percentage measures failure frequency.'),
    ),
    (
        ('Prefer a larger total sum of money paid in the initial transaction to acquire one unit excluding every later use payment.', 'Prefer a smaller total sum of money paid in the initial transaction to acquire one unit excluding every later use payment.', 'larger', 'smaller', 'The full initial acquisition amount is measured; later use payments are excluded.'),
        ('Choose a larger total amount due for the full ownership-transfer transaction for one item excluding every later operating payment.', 'Choose a smaller total amount due for the full ownership-transfer transaction for one item excluding every later operating payment.', 'larger', 'smaller', 'Only the total ownership-transfer transaction amount is measured.'),
        ('Prefer a larger total amount paid for the complete initial acquisition transaction of one item apart from all later use payments.', 'Prefer a smaller total amount paid for the complete initial acquisition transaction of one item apart from all later use payments.', 'larger', 'smaller', 'The full initial acquisition amount is separated from later use bills.'),
        ('Choose a larger total cash amount required to complete the single acquisition transaction for one item excluding later operating payments.', 'Choose a smaller total cash amount required to complete the single acquisition transaction for one item excluding later operating payments.', 'larger', 'smaller', 'The complete single-sale amount is compared; operation payments are outside the measure.'),
        ("Prefer a larger full transaction total paid to become the item's owner excluding later maintenance payments.", "Prefer a smaller full transaction total paid to become the item's owner excluding later maintenance payments.", 'larger', 'smaller', 'The complete ownership-acquisition amount excludes later maintenance money.'),
        ('Prefer a larger total payment for the complete initial transaction to take ownership of one item excluding later use payments.', 'Prefer a smaller total payment for the complete initial transaction to take ownership of one item excluding later use payments.', 'larger', 'smaller', 'The complete initial ownership payment is measured separately from post-acquisition use.'),
        ('Prefer a larger total sum paid to complete the initial acquisition transaction for one item excluding all later use payments.', 'Prefer a smaller total sum paid to complete the initial acquisition transaction for one item excluding all later use payments.', 'larger', 'smaller', 'The full amount covers the initial acquisition; use payments are not included.'),
        ('Choose a larger full total paid by the buyer in the initial sale of one item not later payments for operation.', 'Choose a smaller full total paid by the buyer in the initial sale of one item not later payments for operation.', 'larger', 'smaller', 'The buyer payment for the initial sale is distinct from later operation costs.'),
        ('Prefer a larger total amount required for the entire acquisition transaction of one item before use separate from ongoing operating payments.', 'Prefer a smaller total amount required for the entire acquisition transaction of one item before use separate from ongoing operating payments.', 'larger', 'smaller', 'The complete pre-use ownership acquisition amount is compared with ongoing operating expense excluded.'),
        ('Choose a larger total amount due to obtain one unit in the initial transaction excluding money paid after acquisition.', 'Choose a smaller total amount due to obtain one unit in the initial transaction excluding money paid after acquisition.', 'larger', 'smaller', 'The initial amount to obtain the unit is kept separate from all post-acquisition spending.'),
        ('Prefer a larger total amount for the complete initial ownership-transfer transaction of one item excluding all use costs.', 'Prefer a smaller total amount for the complete initial ownership-transfer transaction of one item excluding all use costs.', 'larger', 'smaller', 'The full ownership-transfer transaction is measured; subsequent use costs are excluded.'),
        ('Choose a larger total amount paid to secure ownership of one item in one acquisition transaction apart from later use bills.', 'Choose a smaller total amount paid to secure ownership of one item in one acquisition transaction apart from later use bills.', 'larger', 'smaller', 'The acquisition transaction amount excludes later bills for use.'),
        ('Prefer a larger full total due for the initial ownership transaction of one item excluding later service payments.', 'Prefer a smaller full total due for the initial ownership transaction of one item excluding later service payments.', 'larger', 'smaller', 'The entire initial ownership amount is separate from later service bills.'),
        ('Choose a larger total amount paid at checkout to acquire one item excluding money paid for later operation.', 'Choose a smaller total amount paid at checkout to acquire one item excluding money paid for later operation.', 'larger', 'smaller', 'The complete initial checkout amount is compared separately from operating expense.'),
        ('Prefer a larger total upfront amount due to obtain one item in the initial transaction not later use payments.', 'Prefer a smaller total upfront amount due to obtain one item in the initial transaction not later use payments.', 'larger', 'smaller', 'The full initial acquisition amount is measured instead of use costs.'),
        ('Choose a larger total amount paid in the full ownership-transfer transaction for one item separate from recurring operating payments.', 'Choose a smaller total amount paid in the full ownership-transfer transaction for one item separate from recurring operating payments.', 'larger', 'smaller', 'The complete initial transaction is not a recurring operating payment.'),
    ),
    (
        ('Prefer a larger total of money spent after acquisition to use one item through the same 40 sessions.', 'Prefer a smaller total of money spent after acquisition to use one item through the same 40 sessions.', 'larger', 'smaller', 'Total post-acquisition spending for the same use exposure is compared.'),
        ('Choose a larger total cash amount paid after acquisition to keep one acquired unit in service for the same 12-month period.', 'Choose a smaller total cash amount paid after acquisition to keep one acquired unit in service for the same 12-month period.', 'larger', 'smaller', 'The full continued-use cost is totaled over an equal year.'),
        ('Prefer a larger total amount of money paid to operate one acquired unit during the same 60 hours.', 'Prefer a smaller total amount of money paid to operate one acquired unit during the same 60 hours.', 'larger', 'smaller', 'Total money for operation during equal exposure is measured after ownership begins.'),
        ('Choose a larger total monetary outlay after acquisition for all 25 matched uses of one item.', 'Choose a smaller total monetary outlay after acquisition for all 25 matched uses of one item.', 'larger', 'smaller', 'The post-acquisition total is compared over the same 25 uses.'),
        ('Prefer a larger overall amount spent to run one acquired unit during the same ten weeks after acquisition.', 'Prefer a smaller overall amount spent to run one acquired unit during the same ten weeks after acquisition.', 'larger', 'smaller', 'Total running expense after acquisition is measured over ten weeks.'),
        ('Choose a larger total sum of all payments for continued use of one already acquired item over the same 12 months.', 'Choose a smaller total sum of all payments for continued use of one already acquired item over the same 12 months.', 'larger', 'smaller', 'All payments for continued use are totaled over the same year.'),
        ('Prefer a larger total amount spent after acquisition to keep using the same item across 30 matched tasks.', 'Prefer a smaller total amount spent after acquisition to keep using the same item across 30 matched tasks.', 'larger', 'smaller', 'The full post-acquisition use expense covers the same 30 tasks.'),
        ('Choose a larger total amount spent after acquisition to keep one item usable throughout the same six-month interval.', 'Choose a smaller total amount spent after acquisition to keep one item usable throughout the same six-month interval.', 'larger', 'smaller', 'All continued-use spending is counted over six months.'),
        ('Prefer a larger total of money paid to operate one owned unit for the same 40 operating hours.', 'Prefer a smaller total of money paid to operate one owned unit for the same 40 operating hours.', 'larger', 'smaller', 'The entire operating expense is compared at equal post-acquisition exposure.'),
        ('Choose a larger total operating outlay for one acquired item during the same 21-day period after acquisition.', 'Choose a smaller total operating outlay for one acquired item during the same 21-day period after acquisition.', 'larger', 'smaller', 'All continued-use spending is totaled for the same three-week period.'),
        ('Prefer a larger total amount paid to continue using one acquired unit during the same eight-week period.', 'Prefer a smaller total amount paid to continue using one acquired unit during the same eight-week period.', 'larger', 'smaller', 'The complete post-acquisition amount covers an equal eight-week interval.'),
        ('Choose a larger sum of post-acquisition money spent to run the same unit through 48 operating cycles.', 'Choose a smaller sum of post-acquisition money spent to run the same unit through 48 operating cycles.', 'larger', 'smaller', 'The full sum for operation is compared across the same 48 cycles.'),
        ('Prefer a larger total amount of money spent to keep one acquired item in use during the same 20 operating days.', 'Prefer a smaller total amount of money spent to keep one acquired item in use during the same 20 operating days.', 'larger', 'smaller', 'All continued-use spending is measured over the same 20 days.'),
        ('Choose a larger total sum paid for operation of one acquired unit over the same fixed 100 kilometers.', 'Choose a smaller total sum paid for operation of one acquired unit over the same fixed 100 kilometers.', 'larger', 'smaller', 'Total post-acquisition operating money covers equal travel distance.'),
        ('Prefer a larger total sum of money spent after acquisition to keep using one unit throughout the same calendar quarter.', 'Prefer a smaller total sum of money spent after acquisition to keep using one unit throughout the same calendar quarter.', 'larger', 'smaller', 'The full continued-use amount is compared over the same quarter.'),
        ('Choose a larger total operating cost paid after acquisition across the same 16-week use interval.', 'Choose a smaller total operating cost paid after acquisition across the same 16-week use interval.', 'larger', 'smaller', 'All post-acquisition operating money is totaled over 16 weeks.'),
    ),
    (
        ('Prefer fewer manual setup steps the user performs to ready one unit for the same task across 30 sessions excluding money.', 'Prefer more manual setup steps the user performs to ready one unit for the same task across 30 sessions excluding money.', 'fewer', 'more', 'Setup actions performed by the user for the same task over 30 sessions quantify effort; payments are excluded.'),
        ('Choose fewer button presses the user makes during each of 24 identical task cycles excluding monetary payments.', 'Choose more button presses the user makes during each of 24 identical task cycles excluding monetary payments.', 'fewer', 'more', 'User actions are counted over the same 24 cycles; money is outside the comparison.'),
        ('Prefer fewer decisions the operator must make to finish each of 20 identical processes excluding spending.', 'Prefer more decisions the operator must make to finish each of 20 identical processes excluding spending.', 'fewer', 'more', 'Required user decisions measure effort for the same 20 processes; they are not monetary expense.'),
        ('Choose less physical effort from the user for one fixed operation in each of 40 matched cycles ignoring money.', 'Choose more physical effort from the user for one fixed operation in each of 40 matched cycles ignoring money.', 'less', 'more', 'User exertion is compared for the same operation and 40 cycles; monetary cost is not part of the measure.'),
        ('Prefer fewer settings the user must enter before running the same item for 18 jobs excluding payment.', 'Prefer more settings the user must enter before running the same item for 18 jobs excluding payment.', 'fewer', 'more', 'Manual setup entries are user work before the same 18 jobs; payment is excluded.'),
        ('Choose fewer personal actions to start one fixed operation across 40 repeated starts excluding money.', 'Choose more personal actions to start one fixed operation across 40 repeated starts excluding money.', 'fewer', 'more', 'Actions required from the person to start the same operation are counted over 40 starts independently of money.'),
        ('Prefer less lifting effort required from the operator to position the same item for 12 identical jobs excluding charges.', 'Prefer more lifting effort required from the operator to position the same item for 12 identical jobs excluding charges.', 'less', 'more', 'Physical effort required of the operator for the same positioning task is compared across 12 jobs; charges are excluded.'),
        ('Choose fewer hand adjustments the user makes to finish the same task over 30 matched sessions excluding payments.', 'Choose more hand adjustments the user makes to finish the same task over 30 matched sessions excluding payments.', 'fewer', 'more', 'Required hands-on adjustments measure user action for the same task and sessions; no payments enter the comparison.'),
        ('Prefer fewer choices the person must make during each of 16 identical tasks leaving money aside.', 'Prefer more choices the person must make during each of 16 identical tasks leaving money aside.', 'fewer', 'more', 'The number of user decisions is fixed to a common task and 16 repetitions; money is irrelevant.'),
        ('Choose less manual entry by the user to configure one unit before each of 22 identical runs excluding monetary charges.', 'Choose more manual entry by the user to configure one unit before each of 22 identical runs excluding monetary charges.', 'less', 'more', 'Manual configuration work is measured before equal runs; no payment enters the comparison.'),
        ('Prefer fewer preparation actions the person performs before the same 32 sessions excluding spending.', 'Prefer more preparation actions the person performs before the same 32 sessions excluding spending.', 'fewer', 'more', 'User preparation actions are compared for a fixed session set; spending is explicitly excluded.'),
        ('Choose fewer connections the operator must make before 14 identical uses separate from money charged.', 'Choose more connections the operator must make before 14 identical uses separate from money charged.', 'fewer', 'more', 'The user performs the connections before the same 14 uses; charges are not counted.'),
        ('Prefer a smaller amount of physical exertion from the user during the same task in 50 equal cycles excluding money.', 'Prefer a larger amount of physical exertion from the user during the same task in 50 equal cycles excluding money.', 'smaller', 'larger', 'User exertion is measured over 50 equal task cycles; monetary amounts are excluded.'),
        ('Choose less hand effort the user applies during each of 25 identical operations excluding payments.', 'Choose more hand effort the user applies during each of 25 identical operations excluding payments.', 'less', 'more', 'Hand effort required from the user for the same 25 operations is compared independently of payments.'),
        ('Prefer fewer cleanup steps the user performs after each of 10 identical tasks excluding all bills.', 'Prefer more cleanup steps the user performs after each of 10 identical tasks excluding all bills.', 'fewer', 'more', 'Required user cleanup actions follow the same tasks; monetary bills are excluded.'),
        ('Choose fewer actions a person must carry out to complete one standard process across 40 matched cases excluding payments.', 'Choose more actions a person must carry out to complete one standard process across 40 matched cases excluding payments.', 'fewer', 'more', 'Required person-actions for the same process and 40 cases measure effort, not money.'),
    ),
)

REJECTED_AMBIGUITIES = (
    ('Prefer fewer bad outcomes.', 'A bad outcome could be low task quality rather than failure to perform the assigned function.'),
    ('Choose a lower total bill.', 'The phrase combines acquisition and ongoing payments without fixing a single monetary field.'),
    ('Prefer less maintenance.', 'This could mean less money, less human work, or fewer failures; it does not isolate one quantity.'),
    ('Choose the faster item.', 'Machine speed and elapsed waiting time are not user effort or setup burden.'),
    ('Prefer a smoother process.', 'Smoothness could refer to output quality, machine operation, or user effort.'),
    ('Choose the option with fewer interruptions.', 'Interruptions may be external pauses rather than failures of intended function.'),
    ('Prefer fewer dollars spent overall.', 'The phrase does not distinguish the initial transaction from later use payments.'),
    ('Choose the easier item.', 'Ease without an explicit user action or effort measure could denote other attributes.'),
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def _semantic_phrases(text):
    return [text[atom['span'][0]:atom['span'][1]] for atom in parse_criterion(text)]


def _load_reference_group(root, source_name, prefix, file_counts):
    rows = []
    lineage = {}
    for filename, expected_count in file_counts.items():
        path = root / filename
        payload = path.read_bytes()
        values = json.loads(payload)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f'Expected {expected_count} phrase-reference rows in {path}')
        for row in values:
            if not isinstance(row.get('id'), str) or not isinstance(row.get('text'), str):
                raise ValueError(f'Phrase-reference row lacks id/text in {path}')
            rows.append(dict(id=f'{prefix}:{row["id"]}', text=row['text']))
        lineage[f'{source_name}/{filename}'] = dict(path=str(path), sha256=_sha(payload), rows=len(values))
    rows.sort(key=lambda row: row['id'])
    return rows, lineage


def _historical_references(protocol):
    old_protocol_path = Path(__file__).with_name('relation_pretrained_protocol.json')
    old_protocol = json.loads(old_protocol_path.read_text(encoding='utf-8'))
    cbf6_root = Path(old_protocol['source_root'])
    cbf7_root = Path(protocol['source_root'])
    cbf6, cbf6_lineage = _load_reference_group(cbf6_root, 'CBF6-source', 'cbf6', CBF6_REFERENCE_COUNTS)
    cbf7, cbf7_lineage = _load_reference_group(cbf7_root, 'CBF7-source', 'cbf7', CBF7_REFERENCE_COUNTS)
    groups = {
        'CBF6_final': cbf6,
        'CBF7_final': cbf7,
    }
    for values in groups.values():
        values.sort(key=lambda row: row['id'])
    lineage = dict(
        CBF6_source_root_protocol=dict(path=str(old_protocol_path), source_root=str(cbf6_root)),
        CBF6_source_files=cbf6_lineage,
        CBF7_source_files=cbf7_lineage,
    )
    return groups, lineage


def _phrase_records(rows, label):
    records = []
    for row in rows:
        full = row['text']
        records.append(dict(group=label, id=row['id'], kind='full', text=full))
        try:
            semantic = _semantic_phrases(full)
        except ValueError as error:
            raise ValueError(f'Frozen compiler rejected {label} phrase {row["id"]}: {error}') from error
        for index, phrase in enumerate(semantic):
            records.append(dict(group=label, id=row['id'], kind=f'semantic_{index}', text=phrase))
    return records


def _mask_direction(text, marker):
    pattern = re.compile(r'(?<![a-z0-9])' + re.escape(marker) + r'(?![a-z0-9])', re.I)
    masked, count = pattern.subn('<DIRECTION>', text, count=1)
    if count != 1 or pattern.search(masked):
        raise ValueError(f'Expected one direction token {marker!r} in {text!r}')
    return masked


def _fresh_atoms():
    if len(MEANING_PAIRS) != len(AXES) or any(len(pairs) != 16 for pairs in MEANING_PAIRS):
        raise ValueError('CBF8 requires sixteen meaning pairs for each of four fields')
    atoms, meaning = [], []
    for axis, pairs in enumerate(MEANING_PAIRS):
        for variant, (positive, negative, positive_marker, negative_marker, rationale) in enumerate(pairs):
            if _mask_direction(positive, positive_marker) != _mask_direction(negative, negative_marker):
                raise ValueError(f'Quantity/exposure changed within pair for field {axis}, variant {variant}')
            pair_id = f'c8-final-atom-pair-{axis}-{variant:02d}'
            ids = {}
            for sign, text in ((1, positive), (-1, negative)):
                weights = [0] * len(AXES)
                weights[axis] = sign
                atom_id = f'c8-final-atom-{axis}-{"higher" if sign > 0 else "lower"}-{variant:02d}'
                atoms.append(dict(id=atom_id, text=text, axis=axis, sign=sign,
                                  pair_id=pair_id, variant=variant, weights=weights,
                                  stratum='final_alias'))
                ids[sign] = atom_id
            meaning.append(dict(pair_id=pair_id, axis=axis, canonical_field=AXES[axis],
                                higher_atom_id=ids[1], lower_atom_id=ids[-1],
                                higher_text=positive, lower_text=negative,
                                quantity_exposure=_mask_direction(positive, positive_marker),
                                rationale=rationale,
                                reversal='Only the comparative quantity token changes; the measured event, quantity, and exposure remain fixed.'))
    return atoms, meaning


def _fresh_compositions(atoms):
    compositions = _coverage_compositions(atoms)
    for row in compositions:
        row['id'] = 'c8-' + row['id']
        row['pair_id'] = 'c8-' + row['pair_id']
    return compositions


def _training_phrases():
    # This API is used only to reject exact collisions with C8 training/validation.
    from schema_support_training_corpus import build_training

    training_atoms, validation_atoms, _ = build_training()
    records = []
    reference_rows = {}
    for label, rows in (('C8_training', training_atoms), ('C8_validation', validation_atoms)):
        reference_rows[label] = []
        for row in rows:
            if not isinstance(row.get('id'), str) or not isinstance(row.get('text'), str):
                raise ValueError(f'{label} builder row lacks id/text')
            reference_rows[label].append(dict(id=row['id'], text=row['text']))
            records.append(dict(group=label, id=row['id'], kind='full', text=row['text']))
            try:
                phrases = _semantic_phrases(row['text'])
            except ValueError as error:
                raise ValueError(f'Frozen compiler rejected {label} phrase {row["id"]}: {error}') from error
            records.extend(dict(group=label, id=row['id'], kind=f'semantic_{index}', text=phrase)
                           for index, phrase in enumerate(phrases))
    reference_bytes = _json_bytes({
        label: sorted(rows, key=lambda row: row['id'])
        for label, rows in reference_rows.items()
    })
    lineage = dict(
        source='Materialized build_training() IDs and full texts; consumed only for exact lexical collision rejection.',
        sha256=_sha(reference_bytes),
        training_atom_rows=len(training_atoms),
        validation_atom_rows=len(validation_atoms),
        phrase_records=len(records))
    return records, lineage


def _audit_compositions(atoms, compositions):
    by_id = {atom['id']: atom for atom in atoms}
    coverage, uses = Counter(), Counter()
    checks = []
    for row in compositions:
        parsed = parse_criterion(row['text'])
        if len(parsed) != 2 or len(row['components']) != 2:
            raise ValueError(f'Composition must have exactly two compiler atoms: {row["id"]}')
        teacher = [0] * len(AXES)
        details = []
        for component, term in zip(row['components'], parsed):
            atom = by_id[component['atom_id']]
            context_lo, context_hi = term['context_span']
            span_lo, span_hi = term['span']
            if not (context_lo <= span_lo < span_hi <= context_hi <= len(row['text'])):
                raise ValueError(f'Invalid compiler span bounds in {row["id"]}')
            if (row['text'][context_lo:context_hi].rstrip('.') != atom['text'].rstrip('.') or
                    term['factor'] != component['factor'] or term['literal_axis'] is not None):
                raise ValueError(f'Compiler decomposition differs from authored component in {row["id"]}')
            teacher[atom['axis']] += atom['sign'] * component['factor']
            uses[atom['id']] += 1
            details.append(dict(atom_id=atom['id'], axis=atom['axis'], sign=atom['sign'],
                                factor=component['factor'], semantic_span=term['span'],
                                context_span=term['context_span']))
        if teacher != row['weights']:
            raise ValueError(f'Composition teacher vector differs from row weights in {row["id"]}')
        coverage[tuple((d['axis'], d['sign'], d['factor']) for d in details)] += 1
        checks.append(dict(id=row['id'], pair_id=row['pair_id'], teacher_components=details,
                           teacher_weights=teacher))
    expected = {tuple(zip(axes, signs, factors))
                for axes in itertools.combinations(range(len(AXES)), 2)
                for signs in itertools.product((-1, 1), repeat=2)
                for factors in itertools.product((1, 2), repeat=2)}
    if set(coverage) != expected or set(uses) != set(by_id) or set(uses.values()) != {2}:
        raise ValueError('Composition support/sign/factor coverage or exact-two atom use failed')
    groups = {}
    for row in compositions:
        groups.setdefault(row['pair_id'], []).append(row)
    if len(groups) != 64 or any(len(rows) != 2 or rows[0]['weights'] != [-w for w in rows[1]['weights']]
                                for rows in groups.values()):
        raise ValueError('Composition full-vector reversal-pair invariant failed')
    return checks, coverage, uses, groups


def _lexical_audit(all_rows, atoms, historical, training_records):
    historical_records = [record for group, rows in historical.items()
                          for record in _phrase_records(rows, group)]
    fresh_records = []
    for row in all_rows:
        fresh_records.append(dict(id=row['id'], kind='full', text=row['text']))
        fresh_records.extend(dict(id=row['id'], kind=f'semantic_{index}', text=phrase)
                             for index, phrase in enumerate(_semantic_phrases(row['text'])))
    full_duplicates = [text for text, count in Counter(_normalized(row['text']) for row in all_rows).items()
                       if count > 1]
    atom_semantic_duplicates = [text for text, count in Counter(
        _normalized(_semantic_phrases(atom['text'])[0]) for atom in atoms).items() if count > 1]
    schema_hits = [dict(id=record['id'], kind=record['kind'], field=field)
                   for record in fresh_records for field in AXES
                   if _contains(_tokens(record['text']), _tokens(field))]
    historical_inclusions = []
    for fresh in fresh_records:
        fresh_tokens = _tokens(fresh['text'])
        for old in historical_records:
            old_tokens = _tokens(old['text'])
            if _contains(fresh_tokens, old_tokens):
                historical_inclusions.append(dict(id=fresh['id'], kind=fresh['kind'],
                                                   reference_group=old['group'], source_id=old['id'],
                                                   reference_kind=old['kind'], phrase=old['text']))
    training_collisions = []
    training_keys = {}
    for record in training_records:
        training_keys.setdefault(_normalized(record['text']), []).append(record)
    for fresh in fresh_records:
        for old in training_keys.get(_normalized(fresh['text']), ()):
            training_collisions.append(dict(id=fresh['id'], kind=fresh['kind'],
                                            reference_group=old['group'], source_id=old['id'],
                                            reference_kind=old['kind']))
    if (full_duplicates or atom_semantic_duplicates or schema_hits or historical_inclusions or
            training_collisions):
        raise ValueError('CBF8 final lexical exclusion failed: ' + repr(dict(
            normalized_full_duplicates=full_duplicates,
            normalized_atom_semantic_duplicates=atom_semantic_duplicates,
            schema_name_hits=schema_hits, historical_phrase_inclusions=historical_inclusions,
            C8_training_validation_exact_collisions=training_collisions)))
    overlap = []
    for row in all_rows:
        overlap.append(dict(id=row['id'], stratum=row['stratum'],
                            **{group: _overlap(row['text'], rows) for group, rows in historical.items()}))
    return dict(
        normalization='Casefold; ASCII alphanumeric tokens; punctuation and whitespace ignored.',
        normalized_full_text_duplicates=full_duplicates,
        normalized_atomic_semantic_duplicates=atom_semantic_duplicates,
        literal_schema_name_hits=schema_hits,
        complete_historical_phrase_inclusions=historical_inclusions,
        exact_C8_training_validation_phrase_collisions=training_collisions,
        historical_phrase_scope='All full texts and frozen-compiler semantic spans in the CBF6 final and CBF7 final phrase references only.',
        C8_collision_scope='Exact normalized full texts and compiler-extracted semantic phrases from C8 training and validation; build_training() is used only for this rejection check.',
        token_overlap_definition=dict(content_words='Unique normalized tokens minus the frozen CBF6 stopword set; no stemming.',
                                      ngrams='Contiguous normalized all-word token ngrams, lengths two through five; descriptive only.',
                                      policy='No overlap threshold and no overlap-driven phrase editing.'),
        token_overlaps=overlap)


def build_final():
    """Return deterministic final atom/composition rows and audit in memory only."""
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    if tuple(protocol.get('schema', ())) != tuple(AXES):
        raise ValueError('CBF8 protocol schema differs from the frozen compiler order')
    if protocol.get('field_definitions') != {
            'reliability': 'The proportion of matched-use trials in which the item performs its intended task without a failure. A higher value means fewer failed trials at equal exposure.',
            'purchase expense': 'The money paid to acquire the item at the initial transaction, excluding later payments for using or maintaining the item. A higher value means a larger initial payment.',
            'operating expense': 'The money paid after acquisition to keep using or maintaining the item over the same use period, excluding the initial transaction. A higher value means larger ongoing payments.',
            'convenience': 'Ease of preparing and using the item for the same intended task. A higher value means less user effort, fewer required actions, or less setup burden. Monetary payments are separate attributes.'}:
        raise ValueError('CBF8 field definitions changed from the committed protocol')
    historical, lineage = _historical_references(protocol)
    atoms, meanings = _fresh_atoms()
    compositions = _fresh_compositions(atoms)
    all_rows = atoms + compositions
    if (len(atoms) != 128 or len(compositions) != 128 or
            len({row['id'] for row in all_rows}) != 256 or
            any(not row['id'].startswith('c8-final-') for row in all_rows)):
        raise ValueError('CBF8 corpus counts or ID namespace failed')
    if Counter((row['axis'], row['sign']) for row in atoms) != Counter(
            {(axis, sign): 16 for axis in range(len(AXES)) for sign in (-1, 1)}):
        raise ValueError('CBF8 field/orientation balance failed')
    atom_pairs = {}
    by_id = {row['id']: row for row in atoms}
    for atom in atoms:
        atom_pairs.setdefault(atom['pair_id'], []).append(atom)
        parsed = parse_criterion(atom['text'])
        if len(parsed) != 1 or parsed[0]['factor'] != 1 or parsed[0]['literal_axis'] is not None:
            raise ValueError(f'Final atom is not one nonliteral unit criterion: {atom["id"]}')
        start, end = parsed[0]['span']
        context_start, context_end = parsed[0]['context_span']
        if not (context_start <= start < end <= context_end <= len(atom['text'])):
            raise ValueError(f'Invalid frozen compiler atom spans: {atom["id"]}')
    if len(atom_pairs) != 64 or any(
            len(pair) != 2 or pair[0]['axis'] != pair[1]['axis'] or
            pair[0]['variant'] != pair[1]['variant'] or
            pair[0]['weights'] != [-weight for weight in pair[1]['weights']]
            for pair in atom_pairs.values()):
        raise ValueError('CBF8 atomic full-vector reversal pairs failed')
    composition_checks, coverage, atom_uses, composition_pairs = _audit_compositions(atoms, compositions)
    training_records, training_lineage = _training_phrases()
    lexical = _lexical_audit(all_rows, atoms, historical, training_records)
    lineage['C8_training_validation_phrase_references'] = training_lineage
    audit = dict(
        experiment='CBF-8 Schema-support domain control: fresh final corpus',
        protocol=dict(path=str(PROTOCOL), sha256=_sha(protocol_bytes)),
        independent_author_provenance=dict(
            author='C8FreshCorpus worker agent, independent from the C8 training-corpus author',
            source='Meaning pairs authored for this final corpus; no C8 training wording used as an idea source.',
            labels='Self-authored meanings and rationales are provenance-only; not human annotation or production data.',
            blind_review='Independent blinded meaning review remains required before any model execution.'),
        source_lineage=lineage,
        counts=dict(atoms=len(atoms), compositions=len(compositions),
                    atom_reversal_pairs=len(atom_pairs),
                    composition_reversal_pairs=len(composition_pairs),
                    training_atom_rows=training_lineage['training_atom_rows'],
                    validation_atom_rows=training_lineage['validation_atom_rows'],
                    training_validation_phrase_records=training_lineage['phrase_records']),
        meaning_review=dict(timing='Authored independently before any CBF8 model execution.',
                            scope='Audit-only rationales; no rationale is present in atom/composition model input.',
                            pairs=meanings,
                            rejected_ambiguous_phrases=[dict(text=text, reason=reason)
                                                        for text, reason in REJECTED_AMBIGUITIES]),
        lexical_checks=lexical,
        composition_teacher_checks=composition_checks,
        coverage=[dict(components=[dict(axis=axis, sign=sign, factor=factor)
                                  for axis, sign, factor in key], count=coverage[key])
                  for key in sorted(coverage)],
        atom_composition_uses=dict(sorted(atom_uses.items())),
        output_sha256=dict(final_atoms_json=_sha(_json_bytes(atoms)),
                           final_compositions_json=_sha(_json_bytes(compositions))))
    return atoms, compositions, audit
