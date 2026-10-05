import copy
import gzip
import io
from pathlib import Path
import sqlite3

import pytest

from research.endgame import neutral_qasper_native_compile as compiler
from research.endgame import neutral_qasper_native_verify as verifier


@pytest.fixture
def cfg():
    return verifier.protocol()


@pytest.fixture
def canonical_modules(cfg):
    return verifier.ir_modules(cfg)


def annotation(ordinal=0, **fields):
    result = {"ordinal": ordinal}
    for field in ("yes_no", "unanswerable", "evidence", "extractive_spans", "free_form_answer"):
        result[field + "_present"] = int(field in fields)
        result[field + "_json"] = verifier.canonical(fields[field]) if field in fields else None
    return result


def paper(**fields):
    result = {"paper_id": "synthetic-paper", "source_partition": "train", "group_id": "synthetic-group", "component_id": "synthetic-component"}
    for field in ("title", "abstract", "full_text"):
        result[field + "_json"] = verifier.canonical(fields[field]) if field in fields else None
    result.update({key: value for key, value in fields.items() if key not in ("title", "abstract", "full_text")})
    return result


def question(text="Is the answer documented?", **extra):
    return {"ordinal": 0, "question_id_present": 0, "question_id_json": None,
            "question_json": verifier.canonical({"question": text, **extra})}


@pytest.mark.parametrize("phase", ["final", "calibration", "confirmation", "unused", "", "train ", None])
def test_public_phase_denials_precede_filesystem_and_database_access(monkeypatch, phase):
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden access")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    for call in (lambda: verifier.verify_phase(phase), lambda: compiler.compile_phase(phase), lambda: compiler.open_phase(phase)):
        with pytest.raises((RuntimeError, ValueError)):
            call()


@pytest.mark.parametrize("stream", ["payload", "source_census", "", None])
def test_bad_stream_is_denied_before_any_pins_or_opens(monkeypatch, stream):
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden access")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with pytest.raises((RuntimeError, ValueError)):
        compiler.open_phase("train", stream)


def test_absent_and_null_booleans_never_become_false():
    decoded = verifier.independent_annotations([
        annotation(0), annotation(1, yes_no=None, unanswerable=None),
        annotation(2, yes_no=False, unanswerable=False), annotation(3, yes_no=True, unanswerable=True),
    ])
    native = verifier.independent_boolean(decoded, "yes_no")
    transformed = verifier.independent_boolean(decoded, "unanswerable")
    assert native["counts"] == [1, 1]
    assert native["distribution"] == [.5, .5]
    assert native["observed_raters"] == 2
    assert [rating["present"] for rating in native["ratings"]] == [False, True, True, True]
    assert [rating["observed"] for rating in native["ratings"]] == [False, False, True, True]
    assert [rating["mapped_value"] for rating in transformed["ratings"]] == [None, None, True, False]
    assert native["conflicting_annotation_ordinals"] == [3]
    assert transformed["conflicting_annotation_ordinals"] == [3]


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
@pytest.mark.parametrize("field", ["yes_no", "unanswerable"])
def test_only_literal_boolean_values_are_observed(field, value):
    with pytest.raises(RuntimeError, match="Boolean type"):
        verifier.independent_annotations([annotation(**{field: value})])


@pytest.mark.parametrize("field", ["evidence", "extractive_spans"])
@pytest.mark.parametrize("value", ["answer", [None], [False], [0], {}])
def test_native_arrays_require_strings_without_coercion(field, value):
    with pytest.raises(RuntimeError, match="string-array type"):
        verifier.independent_annotations([annotation(**{field: value})])


def test_source_only_serving_is_invariant_to_source_labels(cfg, canonical_modules):
    ir, renderer = canonical_modules
    source = paper(title="Original title", full_text=[{"section_name": "Original heading", "paragraphs": ["Original passage"]}],
                   figures_and_tables=[{"caption": "Excluded figure"}], answers="Excluded paper labels")
    blocks, missing = verifier.independent_state(source, ir)
    qa = question(answers=[{"answer": "Excluded nested answer"}], gold="Excluded question labels")
    first = [annotation(yes_no=True, unanswerable=False, evidence=["Original passage"], extractive_spans=["passage"], free_form_answer="Excluded free form")]
    second = [annotation(yes_no=False, unanswerable=True, evidence=["Unmatched evidence"], extractive_spans=["No source occurrence"], free_form_answer="Other free form")]
    census, bundles, _ = verifier.independent_question(source, qa, first, cfg, "train", blocks, missing, ir, renderer)
    _, changed, _ = verifier.independent_question(source, qa, second, cfg, "train", blocks, missing, ir, renderer)
    assert [bundle[0] for bundle in bundles] == [bundle[0] for bundle in changed]
    assert [block.text for block in blocks] == ["Original title", "Original heading", "Original passage"]
    for serving, decision, target, provenance in bundles:
        assert set(serving) == {"id", "task", "locale", "state", "question", "candidates"}
        assert set(decision["metadata"]) == {"endpoint", "target_available"}
        assert provenance["free_form_answers"][0]["native_value"] == "Excluded free form"
    assert census["annotation_count"] == 1


def test_unlabelled_questions_keep_three_endpoints_and_full_census(cfg, canonical_modules):
    ir, renderer = canonical_modules
    source = paper()
    blocks, missing = verifier.independent_state(source, ir)
    census, bundles, targets = verifier.independent_question(source, question(), [annotation()], cfg, "train", blocks, missing, ir, renderer)
    assert blocks[0].id == "document" and blocks[0].text == ""
    assert missing["empty_document"] is True
    assert len(bundles) == 3
    assert census["endpoints"]["qasper.yes_no"] == {"id": None, "eligible": False}
    assert all(not bundle[2]["target_available"] for bundle in bundles)
    coverage = verifier._coverage()
    verifier._count_question(coverage, census, targets)
    assert coverage["qasper.yes_no"]["questions"] == 1
    assert coverage["qasper.yes_no"]["missing_fields"] == 1
    assert coverage["qasper.yes_no"]["decisions"] == 0
    assert coverage["qasper.answerability"]["target_unavailable"] == 1


def test_duplicate_evidence_and_unmatched_native_items_are_retained(canonical_modules):
    ir, _ = canonical_modules
    blocks, _ = verifier.independent_state(paper(title="Same", full_text=[{"section_name": "Same", "paragraphs": ["Same"]}]), ir)
    annotations = verifier.independent_annotations([annotation(evidence=["Same", "FLOAT SELECTED: Figure 1", "Same", ""])])
    target, _ = verifier.independent_native_items(annotations, blocks, False)
    ids = ["title", "section:0000:heading", "section:0000:paragraph:0000"]
    assert target["matched_block_ids"] == ids
    assert target["native_item_count"] == 4 and target["unmatched_item_count"] == 2
    matches = target["ratings"][0]["matches"]
    assert matches[0]["block_ids"] == ids and matches[2]["block_ids"] == ids
    assert target["ratings"][0]["unmatched"] == [{"source_index": 1, "text": "FLOAT SELECTED: Figure 1"}, {"source_index": 3, "text": ""}]
    assert target["target_available"] is True


def test_extraction_keeps_overlaps_duplicates_unicode_offsets_and_unmatched(canonical_modules):
    ir, _ = canonical_modules
    blocks, _ = verifier.independent_state(paper(title="ababa", full_text=[{"paragraphs": ["ababa", "éé"]}]), ir)
    annotations = verifier.independent_annotations([annotation(extractive_spans=["aba", "aba", "", "unmatched", "é"], free_form_answer="ababa")])
    target, occurrences = verifier.independent_native_items(annotations, blocks, True)
    expected = [{"block_id": block_id, "start": start, "end": start + length}
                for block_id, positions, length in (("title", [0, 2], 3), ("section:0000:paragraph:0000", [0, 2], 3), ("section:0000:paragraph:0001", [0, 1], 1))
                for start in positions]
    assert occurrences == expected
    assert target["ratings"][0]["matches"][0]["occurrences"] == target["ratings"][0]["matches"][1]["occurrences"]
    assert target["native_item_count"] == 5 and target["unmatched_item_count"] == 2
    assert target["ratings"][0]["unmatched"] == [{"source_index": 2, "text": ""}, {"source_index": 3, "text": "unmatched"}]
    for item in occurrences:
        block = next(block for block in blocks if block.id == item["block_id"])
        assert block.text[item["start"]:item["end"]] in ("aba", "é")


def test_null_empty_and_absent_arrays_have_distinct_observation_masks(canonical_modules):
    ir, _ = canonical_modules
    blocks, _ = verifier.independent_state(paper(title="source"), ir)
    annotations = verifier.independent_annotations([annotation(0), annotation(1, evidence=None), annotation(2, evidence=[])])
    target, _ = verifier.independent_native_items(annotations, blocks, False)
    assert [(rating["present"], rating["observed"]) for rating in target["ratings"]] == [(False, False), (True, False), (True, True)]
    assert target["target_available"] is False and target["native_item_count"] == 0


def test_columnar_missing_fields_keep_source_positions(canonical_modules):
    ir, _ = canonical_modules
    blocks, missing = verifier.independent_state(paper(full_text={"section_name": [None, "Heading"], "paragraphs": [[None, "", "Text"]]}), ir)
    assert [(block.id, block.text) for block in blocks] == [("section:0000:paragraph:0002", "Text"), ("section:0001:heading", "Heading")]
    assert missing["section_fields"][0]["null_paragraph_ordinals"] == [0]
    assert missing["section_fields"][1]["paragraphs_present"] is False


@pytest.mark.parametrize("full_text", [False, "text", [{"paragraphs": "text"}], [{"section_name": 1}], {"paragraphs": "text"}])
def test_malformed_source_text_is_not_silently_filtered(canonical_modules, full_text):
    ir, _ = canonical_modules
    with pytest.raises(RuntimeError):
        verifier.independent_state(paper(full_text=full_text), ir)


def test_exact_artifact_comparison_does_not_equate_bool_and_int():
    stream = io.BytesIO((verifier.canonical({"observed": 0}) + "\n").encode())
    with pytest.raises(RuntimeError, match="Independent reconstruction"):
        verifier._equal_line(stream, {"observed": False}, "targets")


@pytest.fixture
def synthetic_database(tmp_path):
    path = tmp_path / "source.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE source_groups(group_id TEXT,source_id TEXT,pinned_revision TEXT,component_id TEXT,final_split TEXT,rank_sha256 TEXT);
        CREATE TABLE papers(paper_pk INTEGER,paper_id TEXT,source_partition TEXT,title_json TEXT,abstract_json TEXT,full_text_json TEXT,group_id TEXT,question_count INTEGER,annotation_count INTEGER,payload_json TEXT);
        CREATE TABLE questions(question_pk INTEGER,paper_pk INTEGER,ordinal INTEGER,question_id_present INTEGER,question_id_json TEXT,question_json TEXT,annotation_count INTEGER);
        CREATE TABLE annotations(paper_pk INTEGER,question_pk INTEGER,ordinal INTEGER,yes_no_present INTEGER,yes_no_json TEXT,unanswerable_present INTEGER,unanswerable_json TEXT,evidence_present INTEGER,evidence_json TEXT,extractive_spans_present INTEGER,extractive_spans_json TEXT,free_form_answer_present INTEGER,free_form_answer_json TEXT,annotation_json TEXT,answer_json TEXT,worker_id_json TEXT);
    """)
    for index, phase in enumerate(("train", "dev", "confirmation"), 1):
        connection.execute("INSERT INTO source_groups VALUES(?,?,?,?,?,?)", (str(index), "synthetic", "revision", "component" + str(index), phase, str(index)))
        connection.execute("INSERT INTO papers VALUES(?,?,?,?,?,?,?,?,?,?)", (index, "paper" + str(index), phase, '"source"', None, None, str(index), 1, 1, '{}'))
        connection.execute("INSERT INTO questions VALUES(?,?,?,?,?,?,?)", (index, index, 0, 0, None, '{"question":"Original?"}', 1))
        row = annotation(yes_no=True if phase == "train" else "invalid-if-leaked")
        values = [index, index, 0]
        for field in ("yes_no", "unanswerable", "evidence", "extractive_spans", "free_form_answer"):
            values.extend((row[field + "_present"], row[field + "_json"]))
        connection.execute("INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (*values, '{}', '{}', '"private"'))
    connection.commit()
    connection.close()
    return path


def test_fixed_source_reads_do_not_cross_phase(synthetic_database):
    cfg = {"authority": {"database": {"path": str(synthetic_database)}}, "source": {"id": "synthetic", "revision": "revision"}}
    actual = set()
    with verifier._database("train", cfg, actual) as connection:
        assert verifier._source_groups(connection, "train", cfg) == {"1": "component1"}
        papers = list(verifier._papers(connection, "train"))
        assert [item["paper_pk"] for item in papers] == [1]
        questions = list(verifier._questions(connection, "train", papers[0]))
        annotations = verifier._annotations(connection, "train", papers[0], questions[0])
        assert verifier.independent_annotations(annotations)[0]["yes_no"]["native_value"] is True
        assert list(verifier._questions(connection, "train", {"paper_pk": 2})) == []
        assert verifier._annotations(connection, "train", {"paper_pk": 2}, {"question_pk": 2}) == []



def test_source_membership_uses_custody_rank_order_not_lexical_ids(synthetic_database):
    connection = sqlite3.connect(synthetic_database)
    connection.execute("UPDATE source_groups SET final_split='train',rank_sha256='0' WHERE group_id='2'")
    connection.commit()
    connection.close()
    cfg = {"authority": {"database": {"path": str(synthetic_database)}},
           "source": {"id": "synthetic", "revision": "revision"},
           "resources": {"SQLite_cache_MiB": 16},
           "data": {"expected": {"train": {"groups": 2, "components": 2,
                    "ordered_group_ids_sha256": compiler.value_digest(["2", "1"])}}}}
    with verifier._database("train", cfg, set()) as connection:
        assert list(verifier._source_groups(connection, "train", cfg)) == ["2", "1"]
    # The producer must accept the original ranked membership before reading any paper.
    source = compiler._source_papers("train", cfg, set(), float("inf"))
    native_paper, _ = next(source)
    assert native_paper["group_id"] == "1" and native_paper["rank_sha256"] == "1"
    source.close()

@pytest.mark.parametrize("sql", ["SELECT payload_json FROM papers", "SELECT annotation_json FROM annotations", "SELECT answer_json FROM annotations", "SELECT worker_id_json FROM annotations", "UPDATE papers SET paper_id='changed'", "DELETE FROM questions"])
def test_read_only_column_allowlist_denies_unregistered_access(synthetic_database, sql):
    cfg = {"authority": {"database": {"path": str(synthetic_database)}}}
    before = verifier.digest(synthetic_database)
    with verifier._database("train", cfg, set()) as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(sql).fetchall()
    assert verifier.digest(synthetic_database) == before


def test_cross_role_components_and_mismatched_annotation_papers_fail_before_labels(synthetic_database):
    connection = sqlite3.connect(synthetic_database)
    connection.execute("UPDATE source_groups SET component_id='component1' WHERE final_split='dev'")
    connection.execute("UPDATE annotations SET paper_pk=2 WHERE question_pk=1")
    connection.commit()
    connection.close()
    cfg = {"authority": {"database": {"path": str(synthetic_database)}}, "source": {"id": "synthetic", "revision": "revision"}}
    with verifier._database("train", cfg, set()) as connection:
        with pytest.raises(RuntimeError, match="crosses source roles"):
            verifier._source_groups(connection, "train", cfg)
        with pytest.raises(RuntimeError, match="paper identity"):
            verifier._annotations(connection, "train", {"paper_pk": 1}, {"question_pk": 1})


@pytest.fixture
def selector_fixture(tmp_path, monkeypatch):
    directory = tmp_path / "train"
    directory.mkdir()
    expected = {"papers": 0, "questions": 0, "annotations": 0, "groups": 0, "components": 0, "ordered_group_ids_sha256": verifier.value_digest([])}
    cfg = {"data": {"output_root": str(tmp_path), "expected": {"train": expected}}, "authority": {}}
    monkeypatch.setattr(compiler, "metadata", lambda: cfg)
    monkeypatch.setattr(compiler, "_committed", lambda: "synthetic-test-only")
    files = {}
    for name in (*verifier.STREAMS, "source_census"):
        path = directory / (name + ".jsonl.gz")
        rows = [] if name == "source_census" else [{"id": "synthetic"}]
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                for row in rows:
                    stream.write((verifier.canonical(row) + "\n").encode())
        files[name] = {"path": path.name, "sha256": verifier.digest(path), "bytes": path.stat().st_size, "rows": len(rows)}
    manifest = {"schema": "vey.neutral.qasper.native-projection-manifest.v1", "phase": "train", "complete": True,
                "protocol_sha256": compiler.PROTOCOL_SHA256, "compiler_sha256": verifier.digest(Path(compiler.__file__)),
                "verifier_sha256": verifier.digest(compiler.VERIFIER), "authority": {}, "files": {name: files[name] for name in verifier.STREAMS},
                "source_census": files["source_census"], "source_counts": {key: expected[key] for key in ("papers", "questions", "annotations", "groups", "components")},
                "membership": {"ordered_group_ids_sha256": expected["ordered_group_ids_sha256"]},
                "source_database_unchanged": True, "source_rows_committed_code_only": True, "sealed_labels_accessed": False, "model_outputs": 0, "quality_credit": False}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(verifier.canonical(manifest) + "\n", encoding="utf-8")
    receipt = {"schema": "vey.neutral.qasper.native-projection-verification.v1", "phase": "train", "status": "PASS",
               "protocol_sha256": compiler.PROTOCOL_SHA256, "compiler_sha256": manifest["compiler_sha256"],
               "verifier_sha256": manifest["verifier_sha256"], "manifest_sha256": verifier.digest(manifest_path)}
    (directory / "verification_receipt.json").write_text(verifier.canonical(receipt) + "\n", encoding="utf-8")
    return directory, manifest, receipt


@pytest.mark.parametrize("tamper", ["stream", "manifest", "receipt", "marker", "missing_receipt", "compiler_identity", "verifier_identity", "root_escape"])
def test_selector_rejects_current_artifact_or_receipt_tampering(selector_fixture, tamper):
    directory, manifest, receipt = selector_fixture
    assert list(compiler.open_phase("train")) == [{"id": "synthetic"}]
    if tamper == "stream":
        with (directory / "serving.jsonl.gz").open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "manifest":
        with (directory / "manifest.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
    elif tamper == "receipt":
        receipt["status"] = "FAIL"
        (directory / "verification_receipt.json").write_text(verifier.canonical(receipt), encoding="utf-8")
    elif tamper == "marker":
        (directory / "verification_in_progress.json").write_text("{}", encoding="utf-8")
    elif tamper == "missing_receipt":
        (directory / "verification_receipt.json").unlink()
    else:
        changed = copy.deepcopy(manifest)
        if tamper == "root_escape":
            changed["files"]["serving"]["path"] = "../serving.jsonl.gz"
        else:
            changed[("compiler" if tamper == "compiler_identity" else "verifier") + "_sha256"] = "0" * 64
        (directory / "manifest.json").write_text(verifier.canonical(changed), encoding="utf-8")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        list(compiler.open_phase("train"))


def test_failed_verification_replaces_current_pass_without_changing_manifest(selector_fixture, monkeypatch):
    directory, manifest, prior = selector_fixture
    cfg = compiler.metadata()
    monkeypatch.setattr(verifier, "protocol", lambda: cfg)
    monkeypatch.setattr(verifier, "metadata", lambda: cfg)
    monkeypatch.setattr(verifier, "_committed", lambda: "synthetic-test-only")
    monkeypatch.setattr(verifier, "_selector_module", lambda: compiler)

    def failed_population(*args):
        raise RuntimeError("Independent reconstruction differs: targets")

    monkeypatch.setattr(verifier, "_verify_population", failed_population)
    with pytest.raises(RuntimeError, match="Independent reconstruction"):
        verifier.verify_phase("train")
    current = verifier.decode((directory / "verification_receipt.json").read_text(encoding="utf-8"))
    assert current["status"] == "FAIL" and current["manifest_sha256"] == prior["manifest_sha256"]
    assert current["verifier_sha256"] == manifest["verifier_sha256"]
    assert current["failure"]["check"] == "Independent reconstruction differs: targets"
    assert verifier.digest(directory / "manifest.json") == prior["manifest_sha256"]
    assert not (directory / "verification_in_progress.json").exists()
    archived = verifier.decode((directory / "verification_receipt_superseded_v1.json").read_text(encoding="utf-8"))
    assert archived["status"] == "PASS"
    with pytest.raises(RuntimeError, match="PASS receipt"):
        compiler.open_phase("train")


def test_historical_failure_does_not_override_current_bound_pass(selector_fixture):
    directory, _, _ = selector_fixture
    (directory / "verification_receipt_superseded_v1.json").write_text('{"status":"FAIL"}', encoding="utf-8")
    assert list(compiler.open_phase("train")) == [{"id": "synthetic"}]
