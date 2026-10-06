import hashlib
import json

import pytest

from research.endgame import neutral_hh_native_input_census as census


def features(left, right, ordinal=0):
    value = {'chosen': left, 'rejected': right}
    raw = (json.dumps(value, ensure_ascii=False) + '\n').encode('utf-8')
    return census.row_features(value, raw, ordinal), value, raw


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def test_role_swapping_changes_only_native_line_identity():
    left = '\n\nHuman: Choose a route.\n\nAssistant: North'
    right = '\n\nHuman: Choose a route.\n\nAssistant: South'
    first, _, raw = features(left, right, 4)
    swapped, _, _ = features(right, left, 4)
    assert first['source_line_sha256'] == hashlib.sha256(raw).hexdigest()
    assert first['source_line_sha256'] != swapped['source_line_sha256']
    assert {key: value for key, value in first.items() if key != 'source_line_sha256'} == {
        key: value for key, value in swapped.items() if key != 'source_line_sha256'}
    assert first['full_transcript_sha256s'] == sorted([sha(left), sha(right)])
    expected = json.dumps(sorted([sha(left), sha(right)]), separators=(',', ':'))
    assert first['input_pair_sha256'] == sha(expected)


def test_last_assistant_and_multiturn_root_are_exact():
    context = '\n\nHuman: First question\n\nAssistant: Earlier answer\n\nHuman: Follow-up'
    row, _, _ = features(context + '\n\nAssistant: New answer', context + '\n\nAssistant: Alternative')
    assert row['projectable'] is True
    assert row['reason'] is None
    assert row['context_sha256'] == sha(context)
    assert row['context_characters'] == len(context)
    assert row['response_sha256s'] == sorted([sha(' New answer'), sha(' Alternative')])
    assert row['response_characters'] == sorted([len(' New answer'), len(' Alternative')])
    assert row['root_prompt_proxy_sha256'] == sha('first question')


@pytest.mark.parametrize(('left', 'right', 'reason'), [
    ('no marker', '\n\nHuman: Q\n\nAssistant: A', 'missing_last_assistant_marker'),
    ('\n\nHuman: Q\n\nAssistant: A', '\n\nHuman: Q \n\nAssistant: B', 'shared_context_mismatch'),
    ('Human: Q\n\nAssistant: A', 'Human: Q\n\nAssistant: B', 'unexpected_context_prefix'),
    ('\n\nHuman: Q\n\nAssistant: \t\n', '\n\nHuman: Q\n\nAssistant: B', 'blank_response'),
    ('\n\nHuman: Q\n\nAssistant: A', '\n\nHuman: Q\n\nAssistant: A', 'identical_response'),
    ('\n\nHuman: Q\n\nAssistant: ', '\n\nHuman: Different\n\nAssistant: ', 'shared_context_mismatch'),
    ('\n\nHuman: Q\n\nAssistant: ', '\n\nHuman: Q\n\nAssistant: ', 'blank_response'),
])
def test_nonprojectable_rows_retain_identity_and_ordered_reason(left, right, reason):
    row, _, _ = features(left, right)
    assert row['projectable'] is False
    assert row['reason'] == reason
    assert set(row) == {'ordinal', 'source_line_sha256', 'full_transcript_sha256s',
                        'input_pair_sha256', 'projectable', 'reason'}


@pytest.mark.parametrize('raw', [
    b'{', b'\n', b'[]\n', b'null\n',
    b'{"chosen":"a","chosen":"b","rejected":"c"}\n',
    b'{"chosen":"a","rejected":"b","extra":"c"}\n',
    b'{"chosen":"a"}\n', b'{"chosen":1,"rejected":"b"}\n',
    b'{"chosen":null,"rejected":"b"}\n', b'{"chosen":NaN,"rejected":"b"}\n',
    b'{"chosen":"\xff","rejected":"b"}\n',
    b'{"chosen":"\\ud800","rejected":"b"}\n',
])
def test_malformed_native_rows_fail_without_silent_drop(raw):
    with pytest.raises((ValueError, UnicodeError)):
        census.decode_row(raw)


def test_exact_distinct_suffixes_are_not_normalized():
    row, _, _ = features('\n\nHuman: Q\n\nAssistant: A', '\n\nHuman: Q\n\nAssistant: A ')
    assert row['projectable'] is True
    assert row['response_characters'] == [2, 3]
    assert len(set(row['response_sha256s'])) == 2


def test_normalized_root_groups_do_not_merge_exact_contexts():
    counts = census.Counts()
    for ordinal, prompt in enumerate((' ＡＢＣ\tStraße  ', 'abc STRASSE')):
        context = '\n\nHuman:' + prompt
        row, value, _ = features(context + '\n\nAssistant: One', context + '\n\nAssistant: Two', ordinal)
        assert row['root_prompt_proxy_sha256'] == sha('abc strasse')
        counts.add(row, value)
    summary = counts.result()
    assert summary['rows'] == summary['projectable_rows'] == 2
    assert summary['unique_contexts'] == 2
    assert summary['unique_root_prompt_proxies'] == 1
    assert summary['max_root_prompt_proxy_rows'] == 2
    assert summary['unique_input_pairs'] == 2


def test_all_row_denominator_duplicate_pairs_and_maxima():
    counts = census.Counts()
    left, right = '\n\nHuman: Q\n\nAssistant: Long', '\n\nHuman: Q\n\nAssistant: Short'
    records = [features(left, right, 0), features(right, left, 1), features('x' * 100, 'y', 2)]
    for row, value, _ in records:
        counts.add(row, value)
    summary = counts.result()
    assert summary['rows'] == 3
    assert summary['projectable_rows'] == 2
    assert summary['unprojectable_rows'] == 1
    assert summary['projection_reason_counts'] == {'missing_last_assistant_marker': 1, 'projectable': 2}
    assert summary['unique_input_pairs'] == 2
    assert summary['duplicate_input_pair_rows'] == 1
    assert summary['max_transcript_characters'] == 100
    assert summary['max_context_characters'] == len('\n\nHuman: Q')
    assert summary['max_response_characters'] == len(' Short')




def test_native_bytes_and_value_cannot_disagree():
    row, value, raw = features('\n\nHuman: Q\n\nAssistant: One', '\n\nHuman: Q\n\nAssistant: Two')
    value['chosen'] += ' altered'
    with pytest.raises(ValueError):
        census.row_features(value, raw, row['ordinal'])
