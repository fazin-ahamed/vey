"""CBF-4: exact visible facts and adversarial semantic criterion directions."""
import argparse
import hashlib
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from audit import put
from build import canonical, digest
from run import load

PROTOCOL = Path(__file__).with_name('exact_state_protocol.json')
AXES = ('reliability', 'purchase expense', 'operating expense', 'convenience')
DONORS = ('quality', 'cost', 'cost', 'preference')
TRAIN_PAIRS = ((0, 1), (1, 2), (2, 3), (0, 3))
HELD_PAIRS = ((0, 2), (1, 3))
# Each pair explicitly expresses opposite directions in the named exact axis.
ALIASES = (
    (('least likely to fail', 'most likely to fail'),
     ('most dependable option', 'least dependable option'),
     ('lowest chance of breaking down', 'highest chance of breaking down'),
     ('greatest consistency under repeated use', 'poorest consistency under repeated use')),
    (('largest upfront payment', 'smallest upfront payment'),
     ('highest initial acquisition bill', 'lowest initial acquisition bill'),
     ('greatest amount spent to buy it', 'smallest amount spent to buy it'),
     ('biggest cost at checkout', 'smallest cost at checkout')),
    (('highest recurring running bill', 'lowest recurring running bill'),
     ('largest cost to keep it running', 'smallest cost to keep it running'),
     ('greatest ongoing outlay', 'smallest ongoing outlay'),
     ('biggest regular expenditure during use', 'smallest regular expenditure during use')),
    (('least effort for everyday use', 'most effort for everyday use'),
     ('fewest hassles during routine tasks', 'most hassles during routine tasks'),
     ('easiest option to use day to day', 'hardest option to use day to day'),
     ('smallest amount of user effort', 'largest amount of user effort')),
)


def parse(text, axes=AXES):
    fields = text.rstrip('.').split('; ')
    pairs = []
    for field in fields:
        match = re.fullmatch(r'([^:;]+): (\d+) percent', field)
        if match is None:
            raise ValueError(f'Invalid exact fact: {field!r}')
        pairs.append((match[1], int(match[2])))
    facts = dict(pairs)
    if len(facts) != len(pairs) or any(not 0 <= v <= 100 for v in facts.values()):
        raise ValueError('Duplicate or out-of-range exact field')
    if axes is None:
        return facts
    if set(facts) != set(axes):
        raise ValueError('Exact state does not match its four-axis schema')
    return tuple(facts[a] for a in axes)


def ray(weights):
    divisor = math.gcd(*weights)
    assert divisor > 0
    return tuple(w // divisor for w in weights)


def atomic(axis, sign, weight, template, test=False):
    a = AXES[axis]
    if test:
        forms = (f'Choose the option with the {"greatest" if sign > 0 else "least"} {a}.',
                 f'{a.capitalize()} should be as {"high" if sign > 0 else "low"} as possible.',
                 f'{"Higher" if sign > 0 else "Lower"} {a} is the preference.',
                 f'We favor {"larger" if sign > 0 else "smaller"} values of {a}.')
    else:
        forms = (f'{"Maximize" if sign > 0 else "Minimize"} {a}.',
                 f'Prefer {"higher" if sign > 0 else "lower"} {a}.',
                 f'{"Reward" if sign > 0 else "Penalize"} {a}.',
                 f'Seek {"more" if sign > 0 else "less"} {a}.',
                 f'Give a {"positive" if sign > 0 else "negative"} weight to {a}.',
                 f'Rank options by {a} from {"highest to lowest" if sign > 0 else "lowest to highest"}.')
    return forms[template] + (' Assign double weight to this preference.' if weight == 2 else '')


def composed(weights, template, test=False):
    inds = [i for i, w in enumerate(weights) if w]
    assert len(inds) == 2
    a, b = [AXES[i] for i in inds]
    wa, wb = [weights[i] for i in inds]
    hi = lambda w: 'higher' if w > 0 else 'lower'
    more = lambda w: 'more' if w > 0 else 'less'
    reward = lambda w: 'reward' if w > 0 else 'penalize'
    word = lambda w: 'one' if abs(w) == 1 else 'two'
    if test:
        terms = [f'{hi(wa)} {a}', f'{hi(wb)} {b}']
        if abs(wa) == abs(wb):
            equal = f'{terms[0].capitalize()} and {terms[1]} matter equally.'
        else:
            heavy = 0 if abs(wa) > abs(wb) else 1
            equal = f'{terms[heavy].capitalize()} matters twice as much as {terms[1-heavy]}.'
        return (equal,
                f'Choose using {a} ({"reward" if wa > 0 else "penalty"}, weight {word(wa)}) and {b} ({"reward" if wb > 0 else "penalty"}, weight {word(wb)}).',
                f'Aim for {more(wa)} {a} while also wanting {more(wb)} {b}; count the first preference {"once" if abs(wa) == 1 else "twice"} and the second preference {"once" if abs(wb) == 1 else "twice"}.')[template]
    return (
        f'{"Maximize" if wa > 0 else "Minimize"} {a} and {"maximize" if wb > 0 else "minimize"} {b}. Use weights {word(wa)} and {word(wb)} respectively.',
        f'{reward(wa).capitalize()} {a} with weight {word(wa)}, and {reward(wb)} {b} with weight {word(wb)}.',
        f'Prefer {hi(wa)} {a} (weight {word(wa)}) together with {hi(wb)} {b} (weight {word(wb)}).',
        f'Score options by {"adding" if wa > 0 else "subtracting"} {word(wa)} times {a} and {"adding" if wb > 0 else "subtracting"} {word(wb)} times {b}.',
        f'For {a}, {more(wa)} is better with weight {word(wa)}. For {b}, {more(wb)} is better with weight {word(wb)}.',
        f'{reward(wa).capitalize()} {a}; {reward(wb).capitalize()} {b}. Give the first preference a weight of {word(wa)} and the second a weight of {word(wb)}.'
    )[template]


def criteria():
    out = []
    def add(stratum, weights, template, text):
        out.append(dict(id=digest([stratum, weights, template]), stratum=stratum,
                        weights=list(weights), ray=list(ray(weights)), template=template,
                        text=text, semantic_id=digest(ray(weights))))
    for axis, sign, weight in itertools.product(range(4), (-1, 1), (1, 2)):
        w = [0] * 4; w[axis] = sign * weight
        for t in range(6):
            add('train', w, f'atomic/{t}', atomic(axis, sign, weight, t))
        if weight == 1:
            for t in range(4):
                add('polarity', w, f'atomic_test/{t}', atomic(axis, sign, weight, t, True))
                alias = ALIASES[axis][t][0 if sign > 0 else 1]
                add('alias', w, f'alias/{t}', f'Choose the option with the {alias}.')
    for pair, magnitudes, signs in itertools.product(TRAIN_PAIRS, ((1, 1), (2, 1)), itertools.product((-1, 1), repeat=2)):
        w = [0] * 4
        for i, m, s in zip(pair, magnitudes, signs): w[i] = m * s
        for t in range(6): add('train', w, f'composition/{t}', composed(w, t))
    for stratum, pairs, magnitudes in (('composition', TRAIN_PAIRS, ((1, 2),)),
                                     ('held_combination', HELD_PAIRS, ((1, 1), (2, 1), (1, 2)))):
        for pair, mag, signs in itertools.product(pairs, magnitudes, itertools.product((-1, 1), repeat=2)):
            w = [0] * 4
            for i, m, s in zip(pair, mag, signs): w[i] = m * s
            for t in range(3): add(stratum, w, f'composition_test/{t}', composed(w, t, True))
    return out


def states(rows):
    originals = {}
    for row in rows:
        if row['split'] == 'held_family' or row['family'] not in set(DONORS): continue
        sid = row['scenario_id']
        if sid in originals:
            assert originals[sid]['candidates'] == row['candidates']
        originals[sid] = row
    result = []
    for sid, quality in sorted(originals.items()):
        if quality['family'] != 'quality': continue
        split, _, k, rep = sid.split('/')
        donors = {family: originals[f'{split}/{family}/{k}/{rep}'] for family in set(DONORS)}
        texts, fact_vectors = [], []
        for index in range(int(k)):
            parsed = {family: parse(row['candidates'][index], None) for family, row in donors.items()}
            values = tuple(parsed[family][axis] for axis, family in zip(AXES, DONORS))
            text = '; '.join(f'{axis}: {v} percent' for axis, v in zip(AXES, values)) + '.'
            assert parse(text) == values
            texts.append(text); fact_vectors.append(values)
        result.append(dict(id=f'{split}/joined/{k}/{rep}', source_split=split, K=int(k),
                           donors={family: row['scenario_id'] for family, row in sorted(donors.items())},
                           join='candidate index in source scenario', candidates=texts,
                           parsed_facts=fact_vectors))
    return result


def validate(cc, ss):
    train = [c for c in cc if c['stratum'] == 'train']
    texts = [c['text'] for c in cc]
    assert len(texts) == len(set(texts)) == len({c['id'] for c in cc})
    train_rays = {tuple(c['ray']) for c in train}
    train_supports = {tuple(i for i, w in enumerate(c['weights']) if w) for c in train}
    for c in cc:
        assert all(w in (-2, -1, 0, 1, 2) for w in c['weights'])
        if c['stratum'] in ('composition', 'held_combination'):
            assert tuple(c['ray']) not in train_rays
        if c['stratum'] == 'held_combination':
            assert tuple(i for i, w in enumerate(c['weights']) if w) not in train_supports
        if c['stratum'] == 'alias':
            assert not set(' '.join(AXES).split()) & set(re.findall(r'[a-z]+', c['text'].lower()))
    Y = np.array([c['weights'] for c in train])
    assert np.linalg.matrix_rank(Y) == 4
    for axis in range(4): assert set(Y[:, axis]) == {-2, -1, 0, 1, 2}
    candidate_sets = {}
    for state in ss:
        parsed = [parse(t) for t in state['candidates']]
        assert parsed == [tuple(v) for v in state['parsed_facts']]
        assert len(set(parsed)) == state['K']
        for axis in range(4): assert len({v[axis] for v in parsed}) == state['K']
        candidate_sets.setdefault(state['source_split'], set()).update(state['candidates'])
    for a, b in itertools.combinations(candidate_sets, 2): assert not candidate_sets[a] & candidate_sets[b]
    return dict(criteria=dict(Counter(c['stratum'] for c in cc)), states=dict(Counter(s['source_split'] for s in ss)),
                training_output_rank=4, training_rays=len(train_rays), composition_rays_disjoint=True,
                held_pair_supports_disjoint=True, alias_literal_axis_tokens_absent=True,
                exact_parser_roundtrip=True, source_scenario_and_candidate_splits_disjoint=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path); args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text()); root = args.root or Path(protocol['output_root'])
    source = Path(protocol['source_root']); _, _, rows = load(source)
    cc, ss = criteria(), states(rows); checks = validate(cc, ss)
    root.mkdir(parents=True, exist_ok=True)
    payloads = {'criteria.json': canonical(cc) + b'\n', 'states.json': canonical(ss) + b'\n'}
    for name, content in payloads.items(): put(root, name, content)
    manifest = dict(experiment='CBF-4', source='Deterministic join of project-generated CBF-0 exact facts; new authored criteria',
                    license_class='shipping-train', axes=list(AXES), checks=checks,
                    scope='Four-axis controlled semantic-language screen; not arbitrary criteria or Jev parity',
                    source_manifest_sha256=hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    files={name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()})
    put(root, 'corpus_manifest.json', canonical(manifest) + b'\n'); print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
