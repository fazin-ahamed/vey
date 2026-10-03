"""Controlled criterion grammar and exact signed-schema composition for CBF-5."""
import re
from numbers import Integral


AXES = ('reliability', 'purchase expense', 'operating expense', 'convenience')
# These are the CBF-4 lexical cues, not a semantic alias dictionary.
CUES = re.compile(r'\b(maximize|minimize|higher|lower|more|less|reward|penalize|positive|negative|highest|lowest|greatest|least|high|low|larger|smaller|adding|subtracting|penalty)\b', re.I)
NEGATIVE = {'minimize', 'lower', 'less', 'penalize', 'negative', 'lowest', 'least', 'low', 'smaller', 'subtracting', 'penalty'}
NUMBER = {'one': 1, 'two': 2, '1': 1, '2': 2}
WEIGHT = r'(one|two|1|2)'
PREFIX = re.compile(r'(?:Choose the option with the |Choose using |Prefer |Aim for )', re.I)
LITERAL = re.compile(r'\b(' + '|'.join(re.escape(a) for a in AXES) + r')\b', re.I)


class CriterionSyntaxError(ValueError):
    """A criterion lies outside the frozen, explicitly supported grammar."""


def _bounds(text, lo, hi):
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    if lo == hi:
        raise CriterionSyntaxError('Empty preference clause')
    return lo, hi


def _atom(text, lo, hi, factor):
    lo, hi = _bounds(text, lo, hi)
    fragment = text[lo:hi]
    matches = list(LITERAL.finditer(fragment))
    if len(matches) > 1:
        raise CriterionSyntaxError('Unparsed clause contains multiple schema names')
    # Never accept a partial clause and quietly ignore its other preferences.
    syntax_body = PREFIX.sub('', fragment, count=1)
    if re.search(r';|,|\.(?=\s+\S)|\b(?:and|while|together with|also|but|or|with|plus|as well as|along with|unless|whereas|provided|instead of)\b', syntax_body, re.I):
        raise CriterionSyntaxError('Unsupported multi-clause criterion: ' + fragment)
    if re.search(r'\b(?:weight|weights|times|twice|once|equally|count)\b', fragment, re.I):
        raise CriterionSyntaxError('Unsupported weighting syntax: ' + fragment)
    axis = sign = None
    if matches:
        match = matches[0]
        axis = AXES.index(match[0].lower())
        before = list(CUES.finditer(fragment[:match.start()]))
        after = list(CUES.finditer(fragment[match.end():]))
        cue = before[-1][0].lower() if before else (after[0][0].lower() if after else 'positive')
        sign = -1 if cue in NEGATIVE else 1
    else:
        # Only explicit operators factor out. Intrinsic least/most/lowest cues
        # remain in the neural atom and are never interpreted as alias polarity.
        operator = re.match(r'(maximize|minimize|reward|penalize)\s+', fragment, re.I)
        if operator:
            factor *= -1 if operator[1].lower() in NEGATIVE else 1
            lo += operator.end()
            fragment = text[lo:hi]
    # Same prefix and terminal-period policy as CBF-4 semantic_span, applied to
    # the original atomic substring rather than a rewritten/label-selected term.
    prefix = PREFIX.match(fragment)
    span_lo = lo + (prefix.end() if prefix else 0)
    span_hi = lo + len(fragment.rstrip('.'))
    if span_lo >= span_hi:
        raise CriterionSyntaxError('Empty semantic atom')
    return dict(span=[span_lo, span_hi], context_span=[lo, hi], factor=factor,
                literal_axis=axis, literal_sign=sign)


def parse_criterion(text):
    """Return original-text semantic/context spans and exact multipliers.

    The grammar covers atomic CBF-4 questions, its three held composition
    frames, complete-question joins, weighted joins, explicit reward/penalty
    terms, and the supplied natural ``while also keeping`` construction.
    Unsupported conjunctions or weighting syntax raise CriterionSyntaxError.
    """
    if not isinstance(text, str) or not text.strip():
        raise CriterionSyntaxError('Criterion must be a nonempty string')

    def parse(lo, hi, factor=1):
        lo, hi = _bounds(text, lo, hi)
        body = text[lo:hi]
        core = body.rstrip('.')

        def match(pattern):
            return re.fullmatch(pattern, core, re.I)

        def clause(m, group, multiplier=1):
            start, end = m.span(group)
            return parse(lo + start, lo + end, factor * multiplier)

        m = match(r'With weight ' + WEIGHT + r':\s*(.+);\s*with weight ' + WEIGHT + r':\s*(.+)')
        if m:
            return clause(m, 2, NUMBER[m[1].lower()]) + clause(m, 4, NUMBER[m[3].lower()])
        m = match(r'(.+?);\s*also\s+(.+)')
        if m:
            return clause(m, 1) + clause(m, 2)
        m = match(r'(.+?) and (.+?) matter equally')
        if m:
            return clause(m, 1) + clause(m, 2)
        m = match(r'(.+?) matters twice as much as (.+)')
        if m:
            return clause(m, 1, 2) + clause(m, 2)
        m = match(r'Choose using (.+?) \((reward|penalty), weight ' + WEIGHT +
                  r'\) and (.+?) \((reward|penalty), weight ' + WEIGHT + r'\)')
        if m:
            return (clause(m, 1, (1 if m[2].lower() == 'reward' else -1) * NUMBER[m[3].lower()]) +
                    clause(m, 4, (1 if m[5].lower() == 'reward' else -1) * NUMBER[m[6].lower()]))
        m = match(r'Aim for (.+?) while also wanting (.+?); count the first preference (once|twice) and the second preference (once|twice)')
        if m:
            return clause(m, 1, 1 if m[3].lower() == 'once' else 2) + clause(m, 2, 1 if m[4].lower() == 'once' else 2)
        m = match(r'(Prefer .+?) while also keeping (.+)')
        if m:
            return clause(m, 1) + clause(m, 2)
        m = match(r'(reward|penalize) (.+?) with weight ' + WEIGHT +
                  r',? and (reward|penalize) (.+?) with weight ' + WEIGHT)
        if m:
            return (clause(m, 2, (1 if m[1].lower() == 'reward' else -1) * NUMBER[m[3].lower()]) +
                    clause(m, 5, (1 if m[4].lower() == 'reward' else -1) * NUMBER[m[6].lower()]))
        m = match(r'(.+?)\. Assign double weight to this preference')
        if m:
            return clause(m, 1, 2)
        m = match(r'(reward|penalize) (.+?) with weight ' + WEIGHT)
        if m:
            return clause(m, 2, (1 if m[1].lower() == 'reward' else -1) * NUMBER[m[3].lower()])
        # CBF-4's literal atomic "Give a positive/negative weight" sentence
        # expresses polarity, not an unparsed magnitude or a clause join.
        m = match(r'Give a (positive|negative) weight to (.+)')
        if m:
            if not LITERAL.search(m[2]):
                return clause(m, 2, 1 if m[1].lower() == 'positive' else -1)
            start, end = m.span(2)
            atom = _atom(text, lo + start, lo + end, factor)
            atom['literal_sign'] = 1 if m[1].lower() == 'positive' else -1
            # Preserve the original prototype span, including its operator.
            atom['span'] = [lo, lo + len(core)]
            atom['context_span'] = [lo, hi]
            return [atom]
        return [_atom(text, lo, hi, factor)]

    return parse(0, len(text))


def compile_criterion(text, resolve):
    """Compile four integer coefficients; any unresolved atom yields UNKNOWN.

    ``resolve`` receives each nonliteral parsed atom, including its original
    semantic/context spans. Records retain the term and resolution, so equal
    term strings in different contexts remain unambiguous.
    """
    coefficients = [0, 0, 0, 0]
    records = []
    unknown = False
    for atom in parse_criterion(text):
        lo, hi = atom['span']
        term = text[lo:hi]
        if atom['literal_axis'] is None:
            resolution = resolve(atom)
        else:
            resolution = dict(axis=atom['literal_axis'], sign=atom['literal_sign'])
        if resolution is None:
            unknown = True
        else:
            axis, sign = resolution['axis'], resolution['sign']
            if (not isinstance(axis, Integral) or isinstance(axis, bool) or axis not in range(4) or
                    not isinstance(sign, Integral) or isinstance(sign, bool) or sign not in (-1, 1)):
                raise ValueError('Resolver must return a schema axis and a nonzero signed orientation')
            coefficients[axis] += atom['factor'] * sign
        records.append(dict(atom, term=term, resolution=resolution))
    return (None if unknown else coefficients), records
