"""Pre-outcome CBF-7 final meanings and lexical audit; no artifact writes."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from build import canonical
from schema_grounding_compiler import AXES, parse_criterion
from schema_relation_corpus import (
    _compositions as _cbf6_compositions,
    _contains,
    _normalized,
    _overlap,
    _semantic_phrases,
    _STOP,
    _tokens,
)

PROTOCOL = Path(__file__).with_name('relation_pretrained_protocol.json')

# Higher/lower refers to the schema coordinate, not everyday desirability.
MEANING_PAIRS = (
    (
        ('Prefer fewer crashes per thousand hours of use.', 'Prefer more crashes per thousand hours of use.',
         'Crashes are losses of function. Fixed operating exposure makes fewer crashes the higher-reliability direction.'),
        ('Choose a lower incidence of mechanical jams that halt the machine.', 'Choose a higher incidence of mechanical jams that halt the machine.',
         'Only jams that halt the machine are measured; their lower incidence means less frequent functional failure.'),
        ('Prefer a smaller share of startups aborted by internal hardware failure.', 'Prefer a larger share of startups aborted by internal hardware failure.',
         'The share of failed startups measures functional failure probability, not output quality; smaller is higher reliability.'),
        ('Choose fewer uncommanded shutdowns per hundred sessions.', 'Choose more uncommanded shutdowns per hundred sessions.',
         'Uncommanded shutdowns are device-caused losses of function. The fixed session denominator holds exposure constant.'),
        ('Prefer a lower incidence of the unit refusing to start.', 'Prefer a higher incidence of the unit refusing to start.',
         'Refusing to start is failure to perform the intended function; lower incidence is higher reliability.'),
        ('Choose a smaller proportion of uses ending in a system crash.', 'Choose a larger proportion of uses ending in a system crash.',
         'A system crash ends functioning. Its smaller proportion among uses is the higher-reliability direction.'),
        ('Prefer fewer internal failures that force a restart per hundred sessions.', 'Prefer more internal failures that force a restart per hundred sessions.',
         'Internal failures forcing restarts are explicit functional failure events; fewer at fixed exposure means higher reliability.'),
        ('Choose a lower incidence of the mechanism locking up during use.', 'Choose a higher incidence of the mechanism locking up during use.',
         'Locking up is inability to continue functioning during use; lower incidence means fewer failures.'),
        ('Prefer a smaller share of jobs abandoned because the machine stopped functioning.', 'Prefer a larger share of jobs abandoned because the machine stopped functioning.',
         'The cause of abandonment is explicitly loss of machine function, not result quality; a smaller share is higher reliability.'),
        ('Choose fewer electrical failures that shut the unit down per hundred uses.', 'Choose more electrical failures that shut the unit down per hundred uses.',
         'Electrical failures causing shutdown are functional failures; fewer per fixed number of uses is higher reliability.'),
        ('Prefer a lower incidence of the appliance becoming unresponsive.', 'Prefer a higher incidence of the appliance becoming unresponsive.',
         'Becoming unresponsive is losing the ability to function in response to controls; lower incidence is higher reliability.'),
        ('Choose a smaller proportion of cycles terminated by a broken mechanism.', 'Choose a larger proportion of cycles terminated by a broken mechanism.',
         'A broken mechanism terminating a cycle is explicit functional failure; a smaller cycle proportion means higher reliability.'),
        ('Prefer fewer freezes that disable the device per hundred sessions.', 'Prefer more freezes that disable the device per hundred sessions.',
         'Only disabling freezes are measured. Fewer such events at fixed session exposure is higher reliability.'),
        ('Choose a lower incidence of the tool giving out in the middle of a task.', 'Choose a higher incidence of the tool giving out in the middle of a task.',
         'Giving out mid-task is loss of function rather than slow performance; lower incidence is higher reliability.'),
        ('Prefer a smaller share of runs stopped by a failed component.', 'Prefer a larger share of runs stopped by a failed component.',
         'The measured stopping cause is component failure; a smaller proportion of stopped runs means fewer functional failures.'),
        ('Choose fewer no-start failures per hundred power-on attempts.', 'Choose more no-start failures per hundred power-on attempts.',
         'A no-start failure is inability to begin functioning. The fixed power-on denominator makes fewer failures higher reliability.'),
    ),
    (
        ('Prefer a bigger number on the price sticker for buying the unit.', 'Prefer a smaller number on the price sticker for buying the unit.',
         'The sticker amount for buying the unit is acquisition money; bigger means higher purchase expense even if undesirable.'),
        ('Choose a larger amount of money changing hands to close the sale.', 'Choose a smaller amount of money changing hands to close the sale.',
         'Closing the sale pins the money transfer to acquisition; a larger transfer is higher purchase expense.'),
        ('Prefer a heftier upfront payment to secure the item.', 'Prefer a lighter upfront payment to secure the item.',
         'The upfront payment secures acquisition rather than continued use; heftier means higher purchase expense.'),
        ('Choose a higher figure quoted for buying it outright.', 'Choose a lower figure quoted for buying it outright.',
         'The outright-buying quote is acquisition money; a higher quote is higher purchase expense.'),
        ('Prefer a larger payment handed to the vendor at the ownership handover.', 'Prefer a smaller payment handed to the vendor at the ownership handover.',
         'Payment at ownership handover is acquisition spending; a larger payment is higher purchase expense.'),
        ('Choose a higher asking price for buying the item.', 'Choose a lower asking price for buying the item.',
         'The asking price is explicitly for buying, not using, the item; higher is higher purchase expense.'),
        ('Prefer a greater outlay for the ownership transaction.', 'Prefer a smaller outlay for the ownership transaction.',
         'Outlay for the ownership transaction is acquisition money; greater outlay is higher purchase expense.'),
        ('Choose a higher acquisition fee charged by the seller.', 'Choose a lower acquisition fee charged by the seller.',
         'An acquisition fee explicitly denotes money charged to acquire; higher is higher purchase expense.'),
        ('Prefer a larger total cash payment to seal the ownership deal.', 'Prefer a smaller total cash payment to seal the ownership deal.',
         'The total cash payment seals acquisition of ownership; larger is higher purchase expense.'),
        ('Choose a bigger card charge for buying the item itself.', 'Choose a smaller card charge for buying the item itself.',
         'The card charge is for buying the item itself, excluding recurring use charges; bigger is higher purchase expense.'),
        ('Prefer a larger closing payment to complete buying the goods.', 'Prefer a smaller closing payment to complete buying the goods.',
         'The closing payment completes the buying transaction; larger is higher purchase expense.'),
        ('Choose a higher ticket price for taking ownership of the item.', 'Choose a lower ticket price for taking ownership of the item.',
         'The ticket price is money required to acquire ownership; higher is higher purchase expense.'),
        ('Prefer a bigger negotiated buying price for the machine.', 'Prefer a smaller negotiated buying price for the machine.',
         'The negotiated buying price explicitly measures acquisition spending; bigger is higher purchase expense.'),
        ('Choose a greater capital outflow to secure ownership.', 'Choose a smaller capital outflow to secure ownership.',
         'Capital flowing out to secure ownership is acquisition money; greater outflow is higher purchase expense.'),
        ('Prefer a bigger debit from the account for buying it.', 'Prefer a smaller debit from the account for buying it.',
         'The account debit is explicitly for buying rather than continued use; bigger is higher purchase expense.'),
        ('Choose a higher agreed price for becoming its owner.', 'Choose a lower agreed price for becoming its owner.',
         'The agreed price for becoming the owner is acquisition spending; higher is higher purchase expense.'),
    ),
    (
        ('Prefer a heftier fee for every single use.', 'Prefer a lighter fee for every single use.',
         'The fee repeats with every use; heftier per-use fees mean higher operating expense.'),
        ('Choose a bigger monetary drain on the wallet per session of use.', 'Choose a smaller monetary drain on the wallet per session of use.',
         'The monetary drain is per use session, not at acquisition; bigger means higher operating expense.'),
        ('Prefer a pricier subscription renewed monthly for continued use.', 'Prefer a cheaper subscription renewed monthly for continued use.',
         'Monthly renewal for continued use is explicit recurring money; pricier is higher operating expense.'),
        ('Choose a higher electricity bill per hour of running.', 'Choose a lower electricity bill per hour of running.',
         'Electricity bills tied to running hours are use spending; higher per-hour bills mean higher operating expense.'),
        ('Prefer a bigger monetary spend on consumables per session of use.', 'Prefer a smaller monetary spend on consumables per session of use.',
         'Consumable spending is explicitly monetary and per use session; bigger spending is higher operating expense.'),
        ('Choose heavier dues paid every month to keep using it.', 'Choose lighter dues paid every month to keep using it.',
         'Monthly dues to keep using the item recur during use; heavier dues mean higher operating expense.'),
        ('Prefer a higher monetary rate charged per hour of operation.', 'Prefer a lower monetary rate charged per hour of operation.',
         'A monetary charge per operating hour is recurring use money; higher rates mean higher operating expense.'),
        ('Choose a bigger cost for each cycle of use.', 'Choose a smaller cost for each cycle of use.',
         'Cost measured per use cycle recurs with operation; bigger cost means higher operating expense.'),
        ('Prefer a higher monetary tariff on its hours of use.', 'Prefer a lower monetary tariff on its hours of use.',
         'The tariff is monetary and tied to use hours; higher tariffs mean higher operating expense.'),
        ('Choose a larger weekly bill for continued use of the unit.', 'Choose a smaller weekly bill for continued use of the unit.',
         'The bill recurs weekly for continued use rather than ownership acquisition; larger is higher operating expense.'),
        ('Prefer a costlier rental for each session of use.', 'Prefer a cheaper rental for each session of use.',
         'Rental charged per use session is recurring use money; costlier rental is higher operating expense.'),
        ('Choose higher metered monetary fees per unit of use.', 'Choose lower metered monetary fees per unit of use.',
         'Metered monetary fees recur per unit of use; higher fees mean higher operating expense.'),
        ('Prefer a bigger charge for each task it completes.', 'Prefer a smaller charge for each task it completes.',
         'A charge for each completed task recurs with use; bigger per-task charges mean higher operating expense.'),
        ('Choose a pricier license renewal every year of continued use.', 'Choose a cheaper license renewal every year of continued use.',
         'Yearly license renewal is recurring money needed for continued use; pricier renewal is higher operating expense.'),
        ('Prefer a heavier monthly monetary spend on keeping it running.', 'Prefer a lighter monthly monetary spend on keeping it running.',
         'The money is spent monthly to continue running, not to acquire; heavier spending means higher operating expense.'),
        ('Choose a heavier monetary toll each time it gets used.', 'Choose a lighter monetary toll each time it gets used.',
         'The explicitly monetary toll is incurred each use; heavier tolls mean higher operating expense.'),
    ),
    (
        ('Prefer fewer trips back to the controls per completed job.', 'Prefer more trips back to the controls per completed job.',
         'Trips to controls are required user movement per completed job; fewer trips means less effort and higher convenience.'),
        ('Choose fewer button presses to complete a session.', 'Choose more button presses to complete a session.',
         'Button presses are physical user actions. Fewer per completed session means less effort and higher convenience.'),
        ('Prefer less time spent actively supervising the process.', 'Prefer more time spent actively supervising the process.',
         'Active supervision is human attention work, not passive waiting; less supervision time means less effort and higher convenience.'),
        ('Choose fewer settings adjustments needed for each use.', 'Choose more settings adjustments needed for each use.',
         'Settings adjustments are required user work per use; fewer adjustments means higher convenience.'),
        ('Prefer less close attention needed from its user.', 'Prefer more close attention needed from its user.',
         'Required close attention is mental user effort; less attention demanded means higher convenience.'),
        ('Choose fewer turns of the dial needed for each use.', 'Choose more turns of the dial needed for each use.',
         'Required dial turns are physical user inputs per use; fewer turns means less effort and higher convenience.'),
        ('Prefer less muscle power needed to operate it.', 'Prefer more muscle power needed to operate it.',
         'Required muscle power is physical exertion by the user; less exertion means higher convenience.'),
        ('Choose fewer decisions its user must make during operation.', 'Choose more decisions its user must make during operation.',
         'User decisions demanded during operation are cognitive actions; fewer demanded decisions means higher convenience.'),
        ('Prefer less retyping of the same details on every use.', 'Prefer more retyping of the same details on every use.',
         'Retyping details is repeated data-entry work by the user; less retyping means higher convenience.'),
        ('Choose fewer pieces to hook up each time it is used.', 'Choose more pieces to hook up each time it is used.',
         'Hooking up pieces every use is required user setup work; fewer pieces means less effort and higher convenience.'),
        ('Prefer less need to consult the instructions during use.', 'Prefer more need to consult the instructions during use.',
         'Consulting instructions is required cognitive work during use; less consultation means higher convenience.'),
        ('Choose less bodily exertion required for each use.', 'Choose more bodily exertion required for each use.',
         'Required bodily exertion is directly human effort per use; less means higher convenience.'),
        ('Prefer fewer prompts to answer before starting each session.', 'Prefer more prompts to answer before starting each session.',
         'Answering prompts is user interaction work each session; fewer required answers means higher convenience.'),
        ('Choose less scrubbing needed after each use.', 'Choose more scrubbing needed after each use.',
         'Scrubbing after use is required physical cleanup work; less scrubbing means higher convenience.'),
        ('Prefer fewer refilling actions required from the user between runs.', 'Prefer more refilling actions required from the user between runs.',
         'The quantity is required human refilling actions, not consumable money; fewer actions means higher convenience.'),
        ('Choose fewer fields to fill in by hand after each use.', 'Choose more fields to fill in by hand after each use.',
         'Hand-filling fields is required record-entry work after use; fewer fields means less effort and higher convenience.'),
    ),
)

REJECTED_PHRASES = (
    ('Prefer fewer disappointments.', 'Disappointment could concern output quality rather than functional failure.'),
    ('Choose a machine that lasts longer.', 'Unanchored lifespan could concern physical wear or obsolescence instead of functional failure frequency.'),
    ('Prefer fewer factory recalls.', 'Recall frequency depends on manufacturer policy and can concern safety rather than actual functional failures.'),
    ('Choose fewer calibration problems.', 'Calibration can concern output precision rather than loss of function.'),
    ('Prefer a cheaper option.', 'Cheap does not distinguish acquisition money from recurring use money.'),
    ('Choose a lower total cost of ownership.', 'Total ownership cost combines acquisition and use spending rather than one field.'),
    ('Prefer a larger down payment.', 'A down payment is only a financing component and need not be monotone in total acquisition spending.'),
    ('Choose supplies that wear out faster.', 'Wear rate does not uniquely specify monetary spending and may concern failures.'),
    ('Prefer a steeper meter reading.', 'A meter reading may measure consumption rather than money; accepted metered aliases explicitly name monetary fees.'),
    ('Choose less maintenance.', 'Maintenance could mean labor, money, or failure frequency.'),
    ('Prefer faster results.', 'Machine speed is not the amount of required human effort.'),
    ('Choose less hassle.', 'Hassle could mean effort, waiting, faults, or money; accepted aliases name concrete user work.'),
)

REFERENCE_FILES = {
    'development_atoms.json': 32,
    'development_compositions.json': 50,
    'final_atoms.json': 128,
    'final_compositions.json': 128,
    'training_atoms.json': 32,
    'validation_atoms.json': 32,
    'literal_holdout_atoms.json': 32,
    'literal_cases.json': 155,
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _references(source):
    rows, lineage = {}, {}
    for name, count in REFERENCE_FILES.items():
        path = source / name
        payload = path.read_bytes()
        values = json.loads(payload)
        if len(values) != count:
            raise ValueError(f'Expected {count} reference rows in {path}')
        # CBF6 final files are read only as editorial text-exclusion references.
        rows[name] = [dict(id=r['id'], text=r['text']) for r in values]
        lineage[name] = dict(path=str(path), sha256=_sha(payload), rows=len(values))
    groups = {
        'old_CBF4_5': rows['development_atoms.json'] + rows['development_compositions.json'],
        'old_CBF6_final': rows['final_atoms.json'] + rows['final_compositions.json'],
        'training_literals': rows['training_atoms.json'],
        'validation_literals': rows['validation_atoms.json'],
        'held_literals': rows['literal_holdout_atoms.json'] + rows['literal_cases.json'],
    }
    for values in groups.values():
        values.sort(key=lambda r: r['id'])
    return groups, lineage


def _compositions(atoms):
    # Reuse only the pure CBF6 coverage scheduler, never its meaning phrases.
    rows = _cbf6_compositions(atoms)
    for row in rows:
        row['id'] = 'c7-' + row['id']
        row['pair_id'] = 'c7-' + row['pair_id']
    return rows


def build_final():
    """Return authored atoms, compositions and audit without writing artifacts."""
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    source = Path(protocol['source_root'])
    references, lineage = _references(source)
    atoms, meanings = [], []
    if len(MEANING_PAIRS) != 4 or any(len(pairs) != 16 for pairs in MEANING_PAIRS):
        raise ValueError('Every field requires sixteen opposite meaning pairs')
    for axis, pairs in enumerate(MEANING_PAIRS):
        for variant, (higher, lower, rationale) in enumerate(pairs):
            pair_id = f'c7-final-atom-pair-{axis}-{variant:02d}'
            ids = {}
            for sign, text in ((1, higher), (-1, lower)):
                weights = [0] * 4
                weights[axis] = sign
                atom_id = f'c7-final-atom-{axis}-{"higher" if sign == 1 else "lower"}-{variant:02d}'
                atoms.append(dict(id=atom_id, text=text, axis=axis, sign=sign,
                                  weights=weights, pair_id=pair_id, variant=variant, stratum='final_alias'))
                ids[sign] = atom_id
            meanings.append(dict(pair_id=pair_id, axis=axis, canonical_field=AXES[axis],
                                 higher_atom_id=ids[1], lower_atom_id=ids[-1],
                                 higher_text=higher, lower_text=lower, rationale=rationale,
                                 reversal='Only the comparative direction reverses; the measured quantity and exposure remain fixed.'))
    compositions = _compositions(atoms)
    all_rows = atoms + compositions
    by_id = {a['id']: a for a in atoms}
    if (len(atoms) != 128 or len(compositions) != 128 or
            len({r['id'] for r in all_rows}) != 256 or
            any(not r['id'].startswith('c7-final-') for r in all_rows)):
        raise ValueError('CBF7 counts or fresh ID namespace differ from the protocol')
    if Counter((a['axis'], a['sign']) for a in atoms) != Counter(
            {(a, s): 16 for a in range(4) for s in (-1, 1)}):
        raise ValueError('CBF7 field/direction balance failed')

    duplicates = [text for text, count in Counter(_normalized(r['text']) for r in all_rows).items() if count > 1]
    semantic_duplicates = [text for text, count in Counter(
        _normalized(_semantic_phrases(a['text'])[0]) for a in atoms).items() if count > 1]
    schema_hits = [dict(id=r['id'], field=field) for r in all_rows for field in AXES
                   if _contains(_tokens(r['text']), _tokens(field))]
    old_phrases = [(group, r['id'], phrase)
                   for group, values in references.items() for r in values
                   for phrase in [r['text']] + _semantic_phrases(r['text'])]
    inclusions = [dict(id=r['id'], reference_group=group, source_id=source_id, phrase=phrase)
                  for r in all_rows for group, source_id, phrase in old_phrases
                  if _contains(_tokens(r['text']), _tokens(phrase))]
    reference_duplicates = [dict(id=r['id'], reference_group=group, source_id=old['id'])
                            for r in all_rows for group, values in references.items() for old in values
                            if _normalized(r['text']) == _normalized(old['text'])]
    if duplicates or semantic_duplicates or schema_hits or inclusions or reference_duplicates:
        raise ValueError('CBF7 lexical exclusion failed: ' + repr(dict(
            normalized_duplicates=duplicates, semantic_duplicates=semantic_duplicates,
            schema_hits=schema_hits, complete_old_phrase_inclusions=inclusions,
            reference_text_duplicates=reference_duplicates)))

    uses, coverage = Counter(), Counter()
    teacher_checks = []
    for atom in atoms:
        parsed = parse_criterion(atom['text'])
        if len(parsed) != 1 or parsed[0]['factor'] != 1 or parsed[0]['literal_axis'] is not None:
            raise ValueError('Fresh alias is not a nonliteral unit atom: ' + atom['id'])
    for row in compositions:
        parsed = parse_criterion(row['text'])
        if len(parsed) != 2 or len(row['components']) != 2:
            raise ValueError('Composition must invoke exactly two atoms: ' + row['id'])
        teacher, details = [0] * 4, []
        for component, term in zip(row['components'], parsed):
            atom = by_id[component['atom_id']]
            lo, hi = term['context_span']
            if (row['text'][lo:hi].rstrip('.') != atom['text'].rstrip('.') or
                    term['factor'] != component['factor'] or term['literal_axis'] is not None):
                raise ValueError('Frozen compiler decomposition differs from authored components: ' + row['id'])
            teacher[atom['axis']] += atom['sign'] * component['factor']
            uses[atom['id']] += 1
            details.append(dict(atom_id=atom['id'], axis=atom['axis'], sign=atom['sign'], factor=component['factor']))
        if teacher != row['weights']:
            raise ValueError('Teacher composition sum differs from labels: ' + row['id'])
        coverage[tuple((d['axis'], d['sign'], d['factor']) for d in details)] += 1
        teacher_checks.append(dict(id=row['id'], pair_id=row['pair_id'], teacher_components=details, teacher_weights=teacher))
    expected = {tuple(zip(axes, signs, factors))
                for axes in itertools.combinations(range(4), 2)
                for signs in itertools.product((-1, 1), repeat=2)
                for factors in itertools.product((1, 2), repeat=2)}
    if set(coverage) != expected or set(uses) != set(by_id) or set(uses.values()) != {2}:
        raise ValueError('Incomplete field/sign/factor coverage or atom not used exactly twice')
    for group in (atoms, compositions):
        pairs = {}
        for row in group:
            pairs.setdefault(row['pair_id'], []).append(row)
        if len(pairs) != 64 or any(len(rows) != 2 or rows[0]['weights'] != [-w for w in rows[1]['weights']]
                                  for rows in pairs.values()):
            raise ValueError('Meaning-reversal group invariant failed')

    overlap = [dict(id=row['id'], stratum=row['stratum'],
                    **{group: _overlap(row['text'], values) for group, values in references.items()})
               for row in all_rows]
    audit = dict(
        experiment='CBF-7 Relation-pretrained criterion interface',
        license_class='shipping-train',
        provenance=dict(author='CBF7 corpus-owner agent', source_root=str(source),
                        output_root=protocol['output_root'], generator_path=str(Path(__file__)),
                        generator_sha256=_sha(Path(__file__).read_bytes()),
                        protocol_path=str(PROTOCOL), protocol_sha256=_sha(protocol_bytes),
                        CBF6_final_use='Editorial lexical-exclusion reference only; no predictions or scores read.',
                        filtering='Pre-outcome lexical and grammar exclusions only; no model-outcome filtering.'),
        source_lineage=lineage,
        counts=dict(atoms=128, compositions=128, atom_meaning_pairs=64, composition_reversal_pairs=64,
                    reference_groups={group: len(values) for group, values in references.items()}),
        meaning_review=dict(timing='Authored before any CBF7 encoding, training, or model outcomes.',
                            scope='Author reasoning only; independent blind review is required before encoding.',
                            pairs=meanings,
                            rejected_ambiguous_phrases=[dict(text=t, reason=r) for t, r in REJECTED_PHRASES]),
        lexical_checks=dict(normalization='Casefold; ASCII alphanumeric tokens; punctuation and whitespace ignored.',
                            normalized_duplicates=duplicates, normalized_atomic_semantic_duplicates=semantic_duplicates,
                            literal_schema_name_hits=schema_hits, complete_old_phrase_inclusions=inclusions,
                            normalized_reference_text_duplicates=reference_duplicates,
                            old_phrase_scope='All referenced full texts and every frozen-compiler semantic phrase; CBF4/5 development includes supplied natural examples.'),
        composition_teacher_checks=teacher_checks,
        coverage=[dict(components=[dict(axis=a, sign=s, factor=f) for a, s, f in key], count=coverage[key])
                  for key in sorted(coverage)],
        atom_composition_uses=dict(sorted(uses.items())),
        overlap_definition=dict(content_words='Unique normalized tokens minus the recorded CBF6 stopword set; no stemming.',
                                stopwords=sorted(_STOP),
                                ngrams='Contiguous normalized all-word token ngrams of lengths two through five; maximum distinct shared-ngram count per reference question.',
                                ties='First source ID in lexical order.',
                                composition_frames='Frozen compiler join syntax is intentionally shared; its overlap receives no semantic novelty credit.',
                                policy='Descriptive only; no overlap-threshold or post-outcome filtering.'),
        overlap=overlap,
        output_sha256=dict(final_atoms_json=_sha(canonical(atoms) + b'\n'),
                           final_compositions_json=_sha(canonical(compositions) + b'\n')),
    )
    return atoms, compositions, audit
