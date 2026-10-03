"""Unopened CBF-6 final questions and pre-outcome meaning/lexical audit."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
import re

from schema_grounding_compiler import AXES, parse_criterion


SOURCE_ROOT = Path('/home/fazinahamed/Documents/vey-data/decisionmix/d3/cbf/schema-grounding-v1')
LITERAL_ROOT = Path('/home/fazinahamed/Documents/vey-data/decisionmix/d3/cbf/exact-state-v1/grammar-corrected')

# Each entry is (higher-field question, lower-field question, meaning rationale).
# Monetary preference direction is explicit, not inferred from desirability.
MEANING_PAIRS = (
    (
        ('Prefer a lower probability of malfunction.', 'Prefer a higher probability of malfunction.',
         'Malfunction probability measures functional failure; lowering it raises reliability.'),
        ('Choose a smaller fraction of runs ending in malfunction.', 'Choose a larger fraction of runs ending in malfunction.',
         'The fraction of malfunctioning runs is a failure frequency; a smaller fraction means greater reliability.'),
        ('Prefer fewer breakdowns per hundred uses.', 'Prefer more breakdowns per hundred uses.',
         'The equal hundred-use denominator holds exposure fixed; fewer breakdowns means fewer failures.'),
        ('Choose a lower rate of stoppages caused by faults.', 'Choose a higher rate of stoppages caused by faults.',
         'Fault-caused stoppages are failures, not scheduled pauses; their lower rate raises reliability.'),
        ('Prefer a smaller risk of functional failure.', 'Prefer a larger risk of functional failure.',
         'Functional failure risk is directly opposite to reliability; smaller risk is the positive direction.'),
        ('Choose a higher proportion of fault-free sessions.', 'Choose a lower proportion of fault-free sessions.',
         'Fault-free sessions are successful functioning sessions; their greater proportion means fewer failing sessions.'),
        ('Prefer fewer malfunctions across repeated trials.', 'Prefer more malfunctions across repeated trials.',
         'The trials provide repeated exposure; the count of malfunctions reverses the reliability direction.'),
        ('Choose a smaller percentage of attempts ending in a fault.', 'Choose a larger percentage of attempts ending in a fault.',
         'The percentage of fault-ending attempts is a failure proportion; decreasing it increases reliability.'),
        ('Prefer a higher chance of completing a run without malfunction.', 'Prefer a lower chance of completing a run without malfunction.',
         'Successful completion is explicitly conditioned on absence of malfunction; its higher probability is higher reliability.'),
        ('Choose fewer functional faults over equally many uses.', 'Choose more functional faults over equally many uses.',
         'Equal exposure makes the functional-fault counts comparable; fewer faults is higher reliability.'),
        ('Prefer a lower frequency of loss of function.', 'Prefer a higher frequency of loss of function.',
         'Loss of function is failure; a lower event frequency is higher reliability.'),
        ('Choose a larger share of executions reaching completion without failure.', 'Choose a smaller share of executions reaching completion without failure.',
         'The share completing without failure is a success proportion; larger success share means greater reliability.'),
        ('Prefer fewer incidents of the device ceasing to work.', 'Prefer more incidents of the device ceasing to work.',
         'Ceasing to work denotes functional failure; fewer such incidents is the higher-reliability direction.'),
        ('Choose a lower likelihood of loss of function during normal operation.', 'Choose a higher likelihood of loss of function during normal operation.',
         'The stated event is loss of function rather than poor output quality; its lower likelihood raises reliability.'),
        ('Prefer a smaller fraction of trials interrupted by malfunction.', 'Prefer a larger fraction of trials interrupted by malfunction.',
         'Only malfunction-interrupted trials count; decreasing their proportion means fewer failures.'),
        ('Choose a greater percentage of cycles completed without faults.', 'Choose a smaller percentage of cycles completed without faults.',
         'Fault-free completion is the complement of failing cycles; a greater percentage is higher reliability.'),
    ),
    (
        ('Prefer a larger sum paid to become the owner.', 'Prefer a smaller sum paid to become the owner.',
         'Payment to become the owner is acquisition outlay; larger payment is higher purchase expense, regardless of desirability.'),
        ('Choose a higher cash total handed over at the sale.', 'Choose a lower cash total handed over at the sale.',
         'The sale fixes acquisition timing; higher cash transferred means greater purchase expense.'),
        ('Prefer a greater monetary charge for obtaining the item.', 'Prefer a smaller monetary charge for obtaining the item.',
         'Obtaining the item identifies acquisition rather than continued use; greater monetary charge is higher purchase expense.'),
        ('Choose a larger number of dollars paid at the point of sale.', 'Choose a smaller number of dollars paid at the point of sale.',
         'Dollars paid at the sale directly measure acquisition spending; larger quantity is the positive expense direction.'),
        ('Prefer a higher total price for taking ownership.', 'Prefer a lower total price for taking ownership.',
         'The total price for ownership measures acquisition spending; higher total is higher purchase expense.'),
        ('Choose a greater payment demanded to obtain the item.', 'Choose a smaller payment demanded to obtain the item.',
         'The payment is demanded for obtaining the item, not for using it; greater payment means greater acquisition expense.'),
        ('Prefer a larger cash debit at the moment of buying.', 'Prefer a smaller cash debit at the moment of buying.',
         'A debit at buying is money spent on acquisition; larger debit is the higher purchase-expense direction.'),
        ('Choose a higher monetary amount charged on the sale receipt.', 'Choose a lower monetary amount charged on the sale receipt.',
         'The sale receipt records acquisition charges; higher monetary amount is higher purchase expense.'),
        ('Prefer a greater sum surrendered to acquire the item.', 'Prefer a smaller sum surrendered to acquire the item.',
         'Money surrendered to acquire the item is acquisition payment; greater sum increases purchase expense.'),
        ('Choose a larger price tag for a single acquisition.', 'Choose a smaller price tag for a single acquisition.',
         'A single acquisition price is distinct from repeated operation charges; larger price is higher purchase expense.'),
        ('Prefer a higher cash commitment to buy the item outright.', 'Prefer a lower cash commitment to buy the item outright.',
         'Buying outright fixes ownership acquisition; higher committed cash is higher purchase expense.'),
        ('Choose a greater monetary total due on taking possession.', 'Choose a smaller monetary total due on taking possession.',
         'The payment due upon possession is acquisition outlay; greater total increases purchase expense.'),
        ('Prefer a larger one-time charge for ownership transfer.', 'Prefer a smaller one-time charge for ownership transfer.',
         'The one-time ownership-transfer charge is acquisition spending; larger charge is higher purchase expense.'),
        ('Choose a higher sum remitted to the seller for the item.', 'Choose a lower sum remitted to the seller for the item.',
         'The seller receives the payment for the item itself; higher sum is greater purchase expense.'),
        ('Prefer a greater cash payment on completing the sale.', 'Prefer a smaller cash payment on completing the sale.',
         'Sale completion identifies the acquisition transaction; greater cash payment increases purchase expense.'),
        ('Choose a larger monetary deduction for acquiring the item.', 'Choose a smaller monetary deduction for acquiring the item.',
         'The deduction is money spent acquiring the item; larger deduction is higher purchase expense.'),
    ),
    (
        ('Prefer a larger sum paid for each day of operation.', 'Prefer a smaller sum paid for each day of operation.',
         'Payment for each operating day is recurrent use outlay; larger daily sum is higher operating expense.'),
        ('Choose a higher cash outflow every month of active use.', 'Choose a lower cash outflow every month of active use.',
         'Cash outflow repeats monthly during use; greater outflow is higher operating expense, not acquisition cost.'),
        ('Prefer a greater monetary charge for each repeated session.', 'Prefer a smaller monetary charge for each repeated session.',
         'A monetary charge repeated per session is recurrent use spending; greater charge raises operating expense.'),
        ('Choose a larger number of dollars spent per week of use.', 'Choose a smaller number of dollars spent per week of use.',
         'Dollars per week of use measure recurrent spending; a larger dollar quantity is higher operating expense.'),
        ('Prefer a higher total bill for each month of running the item.', 'Prefer a lower total bill for each month of running the item.',
         'The bill repeats each running month; its higher total raises recurrent operating expense.'),
        ('Choose a greater payment needed repeatedly to continue using the item.', 'Choose a smaller payment needed repeatedly to continue using the item.',
         'Repeated payments to continue use explicitly exclude a single acquisition payment; greater payment is higher operating expense.'),
        ('Prefer a larger cash debit for every additional hour of operation.', 'Prefer a smaller cash debit for every additional hour of operation.',
         'Cash debits recur per additional operating hour; larger debits mean higher recurrent operating expense.'),
        ('Choose a higher monetary amount charged for each routine use.', 'Choose a lower monetary amount charged for each routine use.',
         'Monetary charges for each use are recurrent use outlay; a higher amount is higher operating expense.'),
        ('Prefer a greater sum spent repeatedly to sustain operation.', 'Prefer a smaller sum spent repeatedly to sustain operation.',
         'Money spent repeatedly sustaining operation is recurrent outlay; a greater sum raises operating expense.'),
        ('Choose a larger daily bill for using the item.', 'Choose a smaller daily bill for using the item.',
         'A daily bill for use is recurrent monetary outlay; a larger bill is higher operating expense.'),
        ('Prefer a higher cash requirement for every recurring session.', 'Prefer a lower cash requirement for every recurring session.',
         'The cash requirement repeats for every session; higher cash required means greater operating expense.'),
        ('Choose a greater monetary total due per month of continued operation.', 'Choose a smaller monetary total due per month of continued operation.',
         'The total due repeats monthly throughout operation; greater monetary total is higher operating expense.'),
        ('Prefer a larger repeated charge throughout active use.', 'Prefer a smaller repeated charge throughout active use.',
         'The repeated charge is incurred during use; its larger amount raises recurrent operating expense.'),
        ('Choose a higher sum remitted for each additional day of running.', 'Choose a lower sum remitted for each additional day of running.',
         'Money remitted per additional running day is recurrent outlay; higher sums mean higher operating expense.'),
        ('Prefer a greater cash payment every week the item is used.', 'Prefer a smaller cash payment every week the item is used.',
         'The cash payment recurs weekly conditional on use; greater payment is higher operating expense.'),
        ('Choose a larger monetary deduction per cycle of operation.', 'Choose a smaller monetary deduction per cycle of operation.',
         'Money deducted per operating cycle is repeated use outlay; a larger deduction is higher operating expense.'),
    ),
    (
        ('Prefer fewer manual steps to finish a routine task.', 'Prefer more manual steps to finish a routine task.',
         'Required manual steps quantify user work for task completion; fewer steps is greater convenience.'),
        ('Choose a smaller amount of work demanded from the user.', 'Choose a larger amount of work demanded from the user.',
         'Work demanded from the user is personal effort; less demanded work is higher convenience.'),
        ('Prefer a lower burden of hands-on intervention.', 'Prefer a higher burden of hands-on intervention.',
         'Hands-on intervention is active human work; reducing its burden increases convenience.'),
        ('Choose fewer actions required from the person using the item.', 'Choose more actions required from the person using the item.',
         'Required actions are work by the user rather than autonomous activity; fewer actions means greater convenience.'),
        ('Prefer less exertion by the user during normal operation.', 'Prefer more exertion by the user during normal operation.',
         'User exertion is effort during operation; less exertion is the positive convenience direction.'),
        ('Choose a smaller workload for completing the same task.', 'Choose a larger workload for completing the same task.',
         'Holding the completed task fixed makes workload comparable; smaller workload means less effort and greater convenience.'),
        ('Prefer fewer manual inputs needed for each routine task.', 'Prefer more manual inputs needed for each routine task.',
         'Manual inputs are required user actions; fewer needed inputs reduce effort and increase convenience.'),
        ('Choose a lower amount of human effort per completed task.', 'Choose a higher amount of human effort per completed task.',
         'Human effort per completed task directly quantifies user work; lower effort is greater convenience.'),
        ('Prefer fewer interventions demanded of the operator.', 'Prefer more interventions demanded of the operator.',
         'Interventions demanded of the operator are human work; fewer interventions increase convenience.'),
        ('Choose a smaller number of user actions per routine session.', 'Choose a larger number of user actions per routine session.',
         'User-action counts per session measure interaction effort; smaller counts mean greater convenience.'),
        ('Prefer less hands-on work to achieve the same result.', 'Prefer more hands-on work to achieve the same result.',
         'The result is held fixed; less required hands-on work means greater convenience rather than changed task quality.'),
        ('Choose a lower level of exertion required from the person using the item.', 'Choose a higher level of exertion required from the person using the item.',
         'Required exertion is effort by the user; lowering the required level increases convenience.'),
        ('Prefer fewer manual operations imposed on the user.', 'Prefer more manual operations imposed on the user.',
         'Manual operations imposed on the user are required work; fewer operations raise convenience.'),
        ('Choose a smaller amount of personal labor for routine use.', 'Choose a larger amount of personal labor for routine use.',
         'Personal labor during routine use is user effort; a smaller amount means greater convenience.'),
        ('Prefer less effort expended by the user to complete ordinary tasks.', 'Prefer more effort expended by the user to complete ordinary tasks.',
         'The explicit quantity is effort expended by the user; less effort for task completion increases convenience.'),
        ('Choose fewer hands-on steps needed to finish the job.', 'Choose more hands-on steps needed to finish the job.',
         'Hands-on steps needed for completion are required human work; fewer such steps increase convenience.'),
    ),
)

REJECTED_PHRASES = (
    ('Choose fewer unsuccessful runs.', 'Unsuccessful could mean wrong output rather than functional failure; replaced by runs ending in malfunction.'),
    ('Prefer fewer defective performances.', 'Defective performance could mean low output quality; replaced by explicit malfunctions.'),
    ('Choose a lower chance of an unusable result.', 'An unusable result may reflect quality rather than loss of function; replaced by loss of function during normal operation.'),
    ('Prefer fewer unexpected stoppages.', 'Unexpected stoppages could be external interruptions; accepted wording specifies stoppages caused by faults.'),
    ('Choose a smaller upfront burden.', 'Burden could denote money or effort; accepted acquisition aliases explicitly state monetary payment.'),
    ('Prefer a lower maintenance burden.', 'Maintenance burden could mean labor, frequency, or money; accepted recurrent aliases explicitly state cash or monetary charges.'),
    ('Choose a smoother experience.', 'Smoothness is qualitative and may refer to performance; accepted convenience aliases quantify user effort.'),
    ('Prefer less physical or mental effort from the user.', 'The disjunction is outside the frozen atomic grammar; accepted wording uses undivided user effort.'),
)

_WORDS = re.compile(r"[a-z0-9]+")
_STOP = frozenset('a an the prefer choose option with using for of to at on in from by per each every it item is are be as also and while keeping that this its taking'.split())


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _tokens(text):
    return tuple(_WORDS.findall(text.casefold()))


def _normalized(text):
    return ' '.join(_tokens(text))


def _semantic_phrases(text):
    return [text[a['span'][0]:a['span'][1]] for a in parse_criterion(text)]


def _contains(haystack, needle):
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                               for i in range(len(haystack) - len(needle) + 1))


def _ngrams(tokens, n):
    return {tokens[i:i + n] for i in range(len(tokens) - n + 1)}


def _overlap(text, references):
    tokens = _tokens(text)
    content = set(tokens) - _STOP
    best = None
    shared = {n: dict(count=0, source_id=None, phrases=[]) for n in range(2, 6)}
    for reference in references:
        other_tokens = _tokens(reference['text'])
        other = set(other_tokens) - _STOP
        union = content | other
        score = len(content & other) / len(union) if union else 0.0
        if best is None or score > best['score']:
            best = dict(score=score, source_id=reference['id'], source_text=reference['text'],
                        shared_content_words=sorted(content & other))
        for n in shared:
            intersection = _ngrams(tokens, n) & _ngrams(other_tokens, n)
            if len(intersection) > shared[n]['count']:
                shared[n] = dict(count=len(intersection), source_id=reference['id'],
                                 phrases=[' '.join(p) for p in sorted(intersection)])
    return dict(max_content_word_jaccard=best,
                max_shared_contiguous_ngrams={str(n): result for n, result in shared.items()})


def _source_rows():
    aliases_path = SOURCE_ROOT / 'criteria.json'
    literals_path = LITERAL_ROOT / 'criteria.json'
    aliases_bytes = aliases_path.read_bytes()
    literals_bytes = literals_path.read_bytes()
    old = json.loads(aliases_bytes)
    aliases = [r for r in old if r['stratum'] == 'alias']
    natural = [r for r in old if r.get('provenance', {}).get('kind') == 'supplied_natural_example']
    training = [r for r in json.loads(literals_bytes) if r['stratum'] == 'train']
    if len(aliases) != 32 or len(natural) != 2 or len(training) != 288:
        raise ValueError('Expected 32 old aliases, two CBF-5 natural examples, and 288 literal training questions')
    references = sorted(aliases + natural, key=lambda r: r['id'])
    training.sort(key=lambda r: r['id'])
    lineage = dict(old_criteria=dict(path=str(aliases_path), sha256=_sha(aliases_bytes)),
                   literal_criteria=dict(path=str(literals_path), sha256=_sha(literals_bytes)))
    return references, training, lineage


def _compositions(atoms):
    lookup = {(a['axis'], a['sign'], a['variant']): a for a in atoms}
    schedule = []
    for axes in itertools.combinations(range(4), 2):
        for factors in itertools.product((1, 2), repeat=2):
            for second_sign in (1, -1):
                schedule.append((axes, factors, (1, second_sign)))
    # Six pairs x four weight patterns x two reversal classes give 48 groups.
    # These 16 extra groups give each axis eight extra slots (32 total), so
    # sequential variant assignment uses all 16 aliases twice in each axis.
    extra_pairs = ((0, 1), (2, 3), (0, 2), (1, 3))
    for cycle, factors in enumerate(itertools.product((1, 2), repeat=2)):
        for axes in extra_pairs:
            schedule.append((axes, factors, (1, 1 if cycle % 2 == 0 else -1)))
    used = Counter()
    result = []
    for group, (axes, factors, signs) in enumerate(schedule):
        variants = [used[axis] % 16 for axis in axes]
        used.update(axes)
        for reversal in (1, -1):
            members = [lookup[(axis, sign * reversal, variant)]
                       for axis, sign, variant in zip(axes, signs, variants)]
            questions = [a['text'] for a in members]
            if factors == (1, 1):
                text = questions[0].rstrip('.') + '; also ' + questions[1]
            else:
                words = {1: 'one', 2: 'two'}
                text = (f'With weight {words[factors[0]]}: ' + questions[0].rstrip('.') +
                        f'; with weight {words[factors[1]]}: ' + questions[1])
            weights = [0] * 4
            for member, factor in zip(members, factors):
                weights[member['axis']] = member['sign'] * factor
            result.append(dict(id=f'final-composition-{group:02d}-{ "forward" if reversal == 1 else "reverse"}',
                               text=text, weights=weights,
                               components=[dict(atom_id=a['id'], factor=f) for a, f in zip(members, factors)],
                               pair_id=f'final-composition-pair-{group:02d}', stratum='final_composition'))
    return result


def build_final():
    """Build labels/audit in memory; no encoding, fitting, or filesystem writes."""
    references, training, lineage = _source_rows()
    atoms = []
    meaning = []
    for axis, pairs in enumerate(MEANING_PAIRS):
        if len(pairs) != 16:
            raise ValueError('Every axis requires 16 meaning pairs')
        for variant, (positive, negative, rationale) in enumerate(pairs):
            pair_id = f'final-atom-pair-{axis}-{variant:02d}'
            ids = {}
            for sign, text in ((1, positive), (-1, negative)):
                weights = [0] * 4
                weights[axis] = sign
                atom_id = f'final-atom-{axis}-{ "higher" if sign == 1 else "lower"}-{variant:02d}'
                atoms.append(dict(id=atom_id, text=text, axis=axis, sign=sign, pair_id=pair_id,
                                  variant=variant, weights=weights, stratum='final_alias'))
                ids[sign] = atom_id
            meaning.append(dict(pair_id=pair_id, axis=axis, canonical_field=AXES[axis],
                                higher_atom_id=ids[1], lower_atom_id=ids[-1],
                                higher_text=positive, lower_text=negative, rationale=rationale,
                                reversal='Only the quantitative comparison reverses; the measured event, quantity and context remain fixed.'))
    compositions = _compositions(atoms)
    all_rows = atoms + compositions
    by_id = {a['id']: a for a in atoms}
    normalized = Counter(_normalized(r['text']) for r in all_rows)
    duplicates = [text for text, count in normalized.items() if count > 1]
    semantic_duplicates = [text for text, count in Counter(_normalized(_semantic_phrases(a['text'])[0])
                                                         for a in atoms).items() if count > 1]
    schema_hits = [dict(id=r['id'], field=field) for r in all_rows for field in AXES
                   if _contains(_tokens(r['text']), _tokens(field))]
    old_phrases = [(r['id'], phrase) for r in references
                   for phrase in [r['text']] + _semantic_phrases(r['text'])]
    inclusions = [dict(id=r['id'], source_id=source_id, phrase=phrase)
                  for r in all_rows for source_id, phrase in old_phrases
                  if _contains(_tokens(r['text']), _tokens(phrase))]
    from schema_relation_prepare import literal_atoms
    train_phrases = {_normalized(atom['text']) for r in training for atom in literal_atoms(r)}
    train_texts = {_normalized(r['text']) for r in training}
    train_duplicates = [r['id'] for r in all_rows if _normalized(r['text']) in train_texts]
    train_atom_duplicates = [r['id'] for r in atoms
                            if _normalized(_semantic_phrases(r['text'])[0]) in train_phrases]
    if len(atoms) != 128 or len(compositions) != 128 or len({r['id'] for r in all_rows}) != 256:
        raise ValueError('Final corpus count/ID invariant failed')
    if duplicates or semantic_duplicates or schema_hits or inclusions or train_duplicates or train_atom_duplicates:
        raise ValueError('Final freshness invariant failed: ' + repr(dict(
            normalized_duplicates=duplicates, semantic_duplicates=semantic_duplicates,
            schema_name_hits=schema_hits, old_phrase_inclusions=inclusions,
            training_duplicates=train_duplicates, training_atom_duplicates=train_atom_duplicates)))
    composition_audit = []
    coverage = Counter()
    atom_uses = Counter()
    for row in compositions:
        parsed = parse_criterion(row['text'])
        if len(parsed) != 2:
            raise ValueError('A final composition must contain exactly two parsed atoms')
        teacher = [0] * 4
        details = []
        for component, parsed_atom in zip(row['components'], parsed):
            atom = by_id[component['atom_id']]
            lo, hi = parsed_atom['context_span']
            if (row['text'][lo:hi].rstrip('.') != atom['text'].rstrip('.') or
                    parsed_atom['factor'] != component['factor'] or parsed_atom['literal_axis'] is not None):
                raise ValueError('Compiler decomposition differs from the authored composition')
            teacher[atom['axis']] += atom['sign'] * component['factor']
            atom_uses[atom['id']] += 1
            details.append(dict(atom_id=atom['id'], axis=atom['axis'], sign=atom['sign'],
                                factor=component['factor'], rationale='The positive outer factor preserves the authored atomic sign.'))
        if teacher != row['weights']:
            raise ValueError('Composition teacher sum differs from component labels')
        coverage[tuple((d['axis'], d['sign'], d['factor']) for d in details)] += 1
        composition_audit.append(dict(id=row['id'], pair_id=row['pair_id'],
                                      teacher_components=details, teacher_weights=teacher))
    expected_coverage = {tuple((axis, sign, factor) for axis, sign, factor in zip(axes, signs, factors))
                         for axes in itertools.combinations(range(4), 2)
                         for signs in itertools.product((-1, 1), repeat=2)
                         for factors in itertools.product((1, 2), repeat=2)}
    if set(coverage) != expected_coverage or set(atom_uses) != set(by_id):
        raise ValueError('Incomplete axis/sign/weight coverage or an unused final atom')
    groups = {}
    for row in compositions:
        groups.setdefault(row['pair_id'], []).append(row)
    if len(groups) != 64 or any(len(rows) != 2 or rows[0]['weights'] != [-w for w in rows[1]['weights']]
                                for rows in groups.values()):
        raise ValueError('Composition reversal group invariant failed')
    for atom in atoms:
        parsed = parse_criterion(atom['text'])
        if len(parsed) != 1 or parsed[0]['factor'] != 1 or parsed[0]['literal_axis'] is not None:
            raise ValueError('Final alias is not a nonliteral unit atom under the frozen compiler')
    overlap = [dict(id=row['id'], stratum=row['stratum'],
                    old_aliases_and_CBF5_natural=_overlap(row['text'], references),
                    training_literals=_overlap(row['text'], training)) for row in all_rows]
    audit = dict(
        experiment='CBF-6 Schema Relation Encoder', source_lineage=lineage,
        counts=dict(atoms=len(atoms), compositions=len(compositions), atom_meaning_pairs=len(meaning),
                    composition_reversal_pairs=len(groups), old_aliases=32, CBF5_natural_examples=2,
                    training_literals=len(training)),
        meaning_review=dict(timing='Authored before any CBF-6 encoding or model outcomes',
                            scope='Author semantic reasoning only; independent blind review remains a separate preregistered requirement.',
                            pairs=meaning, rejected_ambiguous_phrases=[dict(text=t, reason=r) for t, r in REJECTED_PHRASES]),
        lexical_checks=dict(normalization='Casefold; ASCII alphanumeric tokens; punctuation and whitespace ignored.',
                            normalized_duplicates=duplicates, normalized_atomic_semantic_duplicates=semantic_duplicates,
                            literal_schema_name_hits=schema_hits, complete_old_phrase_inclusions=inclusions,
                            normalized_training_text_duplicates=train_duplicates,
                            normalized_training_atom_duplicates=train_atom_duplicates,
                            old_phrase_scope='Full old aliases/natural examples and every compiler-extracted semantic phrase.'),
        composition_teacher_checks=composition_audit,
        coverage=[dict(components=[dict(axis=a, sign=s, factor=f) for a, s, f in key], count=coverage[key])
                  for key in sorted(coverage)],
        atom_composition_uses=dict(sorted(atom_uses.items())),
        overlap_definition=dict(content_words='Unique normalized tokens minus the recorded stopword set; no stemming.',
                                stopwords=sorted(_STOP),
                                ngrams='Contiguous normalized all-word token ngrams, lengths two through five; maximum shared distinct-ngram count per reference question.',
                                ties='First source ID in lexical order.',
                                policy='Descriptive only; no outcome-based or overlap-threshold filtering.'),
        overlap=overlap,
        output_sha256=dict(final_atoms_json=_sha(_json_bytes(atoms)),
                           final_compositions_json=_sha(_json_bytes(compositions))),
    )
    return atoms, compositions, audit


def prepare(root):
    """Write the frozen final artifacts once, only on the parent's explicit call."""
    root = Path(root)
    paths = [root / name for name in ('final_atoms.json', 'final_compositions.json', 'final_corpus_audit.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Refusing to overwrite any previously prepared final corpus artifact')
    atoms, compositions, audit = build_final()
    root.mkdir(parents=True, exist_ok=True)
    for path, value in zip(paths, (atoms, compositions, audit)):
        with path.open('xb') as stream:
            stream.write(_json_bytes(value))
    return atoms, compositions, audit
