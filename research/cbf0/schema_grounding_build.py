"""Deterministic CBF-5 criterion cases; no candidate states or encoder work."""
import itertools
import math

from build import digest
from schema_grounding_compiler import AXES


LITERAL_STRATA = ('polarity', 'composition', 'held_combination')
LITERAL_EXAMPLES = (
    ('Minimize purchase expense.', (0, -1, 0, 0)),
    ('Reward reliability with weight two and penalize operating expense with weight one.', (2, 0, -1, 0)),
    ('Reliability matters twice as much as low operating expense.', (2, 0, -1, 0)),
)
NATURAL_EXAMPLES = (
    ('Prefer the option least likely to fail while also keeping the regular running bill small.', (1, 0, -1, 0)),
    ('Prefer the option most likely to fail while also keeping the regular running bill large.', (-1, 0, 1, 0)),
)


def ray(weights):
    """The unchanged CBF-4 positive-ray convention, without its torch imports."""
    divisor = math.gcd(*weights)
    if divisor <= 0:
        raise ValueError('Case provenance requires a nonzero teacher direction')
    return tuple(w // divisor for w in weights)


def _gold(weights):
    return [dict(axis=axis, sign=1 if weight > 0 else -1, factor=abs(weight))
            for axis, weight in enumerate(weights) if weight]


def _case(stratum, weights, template, text, gold_atoms, provenance):
    weights = list(weights)
    return dict(id=digest([stratum, weights, template]), text=text, stratum=stratum,
                weights=weights, semantic_id=digest(ray(weights)), ray=list(ray(weights)),
                template=template, gold_atoms=gold_atoms, provenance=provenance,
                text_sha256=digest(text))


def build_cases(source_criteria):
    """Return 32 old aliases, 155 literal tests, and 50 alias compositions.

    Source aliases enter only this generator's gold/provenance lookup. Neither
    that lookup nor teacher vectors are exposed to criterion parsing/inference.
    Existing IDs, wording, weights, templates, and semantic IDs are retained.
    New questions use fixed axis-pair/sign/frame order and canonical hashes.
    """
    source = list(source_criteria)
    if len({c['id'] for c in source}) != len(source):
        raise ValueError('Duplicate source criterion IDs')
    aliases = [c for c in source if c['stratum'] == 'alias']
    literals = [c for c in source if c['stratum'] in LITERAL_STRATA]
    if len(aliases) != 32 or len(literals) != 152:
        raise ValueError('Expected the corrected CBF-4 32 aliases and 152 held literal questions')
    alias_lookup = {}
    for c in aliases:
        weights = c['weights']
        atoms = _gold(weights)
        if len(weights) != len(AXES) or len(atoms) != 1 or atoms[0]['factor'] != 1:
            raise ValueError('Source aliases must be unit signed schema atoms')
        axis, sign = atoms[0]['axis'], atoms[0]['sign']
        template = c['template']
        if template not in ('alias/0', 'alias/1', 'alias/2', 'alias/3'):
            raise ValueError('Unexpected source alias variant')
        key = (axis, sign, int(template.rsplit('/', 1)[1]))
        if key in alias_lookup:
            raise ValueError('Duplicate source alias axis/sign/variant')
        alias_lookup[key] = c
    expected = set(itertools.product(range(4), (-1, 1), range(4)))
    if set(alias_lookup) != expected:
        raise ValueError('Incomplete source alias directions/variants')

    cases = []
    for c in source:
        if c['stratum'] not in ('alias',) + LITERAL_STRATA:
            continue
        old = dict(c)
        old.update(stratum='alias' if c['stratum'] == 'alias' else 'literal',
                   gold_atoms=_gold(c['weights']), text_sha256=digest(c['text']),
                   provenance=dict(kind='CBF4_corrected', source_id=c['id'],
                                   source_stratum=c['stratum'], source_sha256=digest(c)))
        cases.append(old)
    for index, (text, weights) in enumerate(LITERAL_EXAMPLES):
        cases.append(_case('literal', weights, f'literal_example/{index}', text, _gold(weights),
                           dict(kind='supplied_literal_example', example=index)))

    for pair in itertools.combinations(range(4), 2):
        for signs in itertools.product((-1, 1), repeat=2):
            for frame in range(2):
                components = [alias_lookup[(axis, sign, frame)] for axis, sign in zip(pair, signs)]
                questions = [c['text'] for c in components]
                factors = (1, 1) if frame == 0 else (2, 1)
                if frame == 0:
                    text = questions[0].rstrip('.') + '; also ' + questions[1]
                else:
                    text = 'With weight two: ' + questions[0].rstrip('.') + '; with weight one: ' + questions[1]
                weights = [0] * 4
                gold = []
                for axis, sign, factor in zip(pair, signs, factors):
                    weights[axis] += sign * factor
                    gold.append(dict(axis=axis, sign=sign, factor=factor))
                provenance = dict(kind='fixed_alias_join', axes=list(pair), signs=list(signs),
                                  frame=frame, source_ids=[c['id'] for c in components],
                                  source_sha256=[digest(c) for c in components])
                cases.append(_case('alias_composition', weights, f'alias_join/{frame}', text, gold, provenance))
    for index, (text, weights) in enumerate(NATURAL_EXAMPLES):
        cases.append(_case('alias_composition', weights, f'natural_example/{index}', text,
                           _gold(weights), dict(kind='supplied_natural_example', example=index)))

    if len(cases) != 237 or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Frozen case count or unique-ID invariant failed')
    if len({c['text'] for c in cases}) != len(cases):
        raise ValueError('Frozen questions contain duplicate visible strings')
    return cases
