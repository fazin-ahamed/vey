"""CBF-5 train-only signed prototypes and fixed template-nuisance projection."""

import numpy as np


AXES = ('reliability', 'purchase expense', 'operating expense', 'convenience')
TEMPLATES = tuple(f'atomic/{i}' for i in range(6))
METHODS = ('cosine_positive', 'asg', 'asg_projected', 'cosine_signed',
           'cosine_positive_projected', 'cosine_signed_projected')
DIMENSION = 384
PROJECTED_ZERO_CUTOFF = np.finfo(np.float64).eps * DIMENSION


def _unit(value):
    """Normalize finite nonzero vectors without overflow or underflow."""
    if not np.all(np.isfinite(value)):
        return None
    scale = np.max(np.abs(value), axis=-1, keepdims=True)
    if np.any(scale == 0):
        return None
    scaled = value / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def fit_prototypes(H, training_records):
    """Fit eight normalized means and all nonzero training-template contrasts.

    Feature rows must match the supplied record order. Only the 48 unit-weight
    training atoms are accepted; their labels never enter query resolution.
    """
    records = list(training_records)
    if len(records) != 48:
        raise ValueError('Prototype fitting requires exactly 48 training atoms')
    features = np.asarray(H, dtype=np.float64)
    if features.shape != (48, DIMENSION):
        raise ValueError(f'Training features must have shape (48, {DIMENSION})')
    features = _unit(features)
    if features is None:
        raise ValueError('Training features must be finite nonzero vectors')

    members = {}
    ids, texts, membership = set(), set(), []
    for row, record in enumerate(records):
        if record.get('stratum') != 'train':
            raise ValueError('Held or non-training records cannot fit prototypes')
        template = record.get('template')
        if template not in TEMPLATES:
            raise ValueError('Prototype members must use the six atomic templates')
        weights = np.asarray(record.get('weights'), dtype=np.float64)
        if weights.shape != (4,) or not np.all(np.isfinite(weights)):
            raise ValueError('Training weights must be finite four-axis vectors')
        nonzero = np.flatnonzero(weights)
        if len(nonzero) != 1 or abs(weights[nonzero[0]]) != 1:
            raise ValueError('Prototype members must be signed unit-weight atoms')
        axis = int(nonzero[0])
        sign = int(weights[axis])
        prototype = 2 * axis + (0 if sign > 0 else 1)
        key = (prototype, template)
        if key in members:
            raise ValueError('Every signed class needs one member per template')
        identifier, text = record.get('id'), record.get('text')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('Training membership IDs must be unique nonempty strings')
        if not isinstance(text, str) or not text.strip() or text in texts:
            raise ValueError('Training membership texts must be unique nonempty strings')
        ids.add(identifier)
        texts.add(text)
        members[key] = row
        membership.append(dict(feature_row=row, id=identifier, text=text,
                               template=template, prototype=prototype,
                               axis=axis, sign=sign))

    expected = {(prototype, template) for prototype in range(8) for template in TEMPLATES}
    if set(members) != expected:
        raise ValueError('All four axes, both signs and six templates must be balanced')
    prototype_rows = [[members[(prototype, template)] for template in TEMPLATES]
                      for prototype in range(8)]
    prototypes = _unit(np.array([features[rows].mean(axis=0) for rows in prototype_rows]))
    if prototypes is None:
        raise ValueError('A signed class has a zero or nonfinite mean prototype')

    template_rows = [[members[(prototype, template)] for prototype in range(8)]
                     for template in TEMPLATES]
    template_means = np.array([features[rows].mean(axis=0) for rows in template_rows])
    contrasts = template_means - template_means.mean(axis=0)
    _, spectrum, vt = np.linalg.svd(contrasts, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(contrasts.shape) * spectrum[0]
    basis = vt[spectrum > cutoff]
    if len(basis) > 5:
        raise ValueError('Six centered template means cannot supply more than five contrasts')

    labels = [dict(prototype=i, axis=i // 2, axis_name=AXES[i // 2],
                   sign=1 if i % 2 == 0 else -1,
                   member_ids=[records[row]['id'] for row in rows])
              for i, rows in enumerate(prototype_rows)]
    metadata = dict(training_count=48, test_count=0, axes=list(AXES),
                    membership=membership, prototype_labels=labels,
                    template_membership=[dict(template=template,
                                              member_ids=[records[row]['id'] for row in rows])
                                         for template, rows in zip(TEMPLATES, template_rows)],
                    singular_values=spectrum.tolist(), rank_cutoff=float(cutoff),
                    basis_rank=int(len(basis)), projection_space='layer1_span',
                    projected_zero_cutoff=float(PROJECTED_ZERO_CUTOFF))
    return prototypes, basis, metadata


def _project_unit(value, basis):
    projected = value - (value @ basis.T) @ basis
    norms = np.linalg.norm(projected, axis=-1)
    # Unit inputs annihilated to FP64 roundoff do not define a cosine direction.
    if not np.all(np.isfinite(norms)) or np.any(norms <= PROJECTED_ZERO_CUTOFF):
        return None
    return _unit(projected)


def resolve(h, prototypes, basis, method):
    """Resolve a feature to a signed schema atom, or None for UNKNOWN.

    Equal evidence uses the smallest schema index; signed retrieval orders the
    positive prototype before the negative prototype within each schema axis.
    Difference-based methods reject exact-zero orientation, without a sign
    fallback or confidence threshold.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown schema-grounding method: {method}')
    query = np.asarray(h, dtype=np.float64)
    base = np.asarray(prototypes, dtype=np.float64)
    if query.shape != (DIMENSION,) or base.shape != (8, DIMENSION):
        raise ValueError(f'Expected query ({DIMENSION},) and prototypes (8, {DIMENSION})')
    query, base = _unit(query), _unit(base)
    if query is None or base is None:
        return None

    if method.endswith('_projected'):
        nuisance = np.asarray(basis, dtype=np.float64)
        if nuisance.ndim != 2 or nuisance.shape[1] != DIMENSION or nuisance.shape[0] > 5:
            raise ValueError(f'Nuisance basis must have shape (rank <= 5, {DIMENSION})')
        if not np.all(np.isfinite(nuisance)):
            return None
        query, base = _project_unit(query, nuisance), _project_unit(base, nuisance)
        if query is None or base is None:
            return None

    cosines = base @ query
    if not np.all(np.isfinite(cosines)):
        return None
    evidence = cosines[::2] - cosines[1::2]
    if method.startswith('cosine_signed'):
        nearest = int(np.argmax(cosines))
        axis, sign = nearest // 2, 1 if nearest % 2 == 0 else -1
    else:
        axis = int(np.argmax(cosines[::2] if method.startswith('cosine_positive')
                             else np.abs(evidence)))
        orientation = evidence[axis]
        if orientation == 0:
            return None
        sign = 1 if orientation > 0 else -1
    return dict(axis=axis, sign=sign, evidence=evidence.tolist(), cosines=cosines.tolist())
