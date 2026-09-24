"""Schema-compiler tests: the ACT safety contract must be exact, not statistical.

These defend the property that makes Vey different from a generative tool
caller: an invalid action cannot be emitted, and a missing required value
becomes ASK_FOR_INFO rather than an invented one.
"""
import pytest

from vey.act import (
    ArgSpec, ToolCard, FieldDecision, compile_call, is_schema_valid, coerce,
    VALUE_STATE, MISSING, AMBIGUOUS,
)
from vey.act.extract import extract_slot


FLIGHT = ToolCard(
    tool_id="book_flight", name="book_flight",
    description="Book a flight for a passenger",
    args=(
        ArgSpec("origin", "string", True, "departure city"),
        ArgSpec("destination", "string", True, "arrival city"),
        ArgSpec("date", "date", True, "travel date"),
        ArgSpec("passengers", "integer", False, "traveler count"),
        ArgSpec("cabin", "enum", False, "cabin class", enum=("economy", "business", "first")),
    ),
)


def test_card_text_is_canonical_and_covers_schema():
    t = FLIGHT.card_text()
    assert "book flight" in t
    assert "required: origin, destination, date" in t
    assert "cabin: enum(economy,business,first)" in t
    # the same card always renders the same text (no ordering leak)
    assert FLIGHT.card_text() == t


def test_rejects_unsupported_arg_type():
    with pytest.raises(ValueError):
        ArgSpec("x", "tensor", True, "bad type")
    with pytest.raises(ValueError):
        ArgSpec("x", "enum", True, "no values")   # enum without values
    with pytest.raises(ValueError):
        ArgSpec("x", "array", True, "no item type")


def test_card_roundtrips_through_dict():
    restored = ToolCard.from_dict(FLIGHT.to_dict())
    assert restored.card_text() == FLIGHT.card_text()
    assert restored.required_paths() == ("origin", "destination", "date")


def test_coerce_rejects_wrong_types_exactly():
    assert coerce(FLIGHT.arg("date"), "2026-09-25") == ("2026-09-25", True)
    assert coerce(FLIGHT.arg("date"), "not a date")[1] is False
    assert coerce(FLIGHT.arg("passengers"), "2")[0] == 2
    assert coerce(FLIGHT.arg("passengers"), "2.5")[1] is False     # not an integer
    assert coerce(FLIGHT.arg("passengers"), "two")[1] is False
    assert coerce(FLIGHT.arg("cabin"), "business")[0] == "business"
    assert coerce(FLIGHT.arg("cabin"), "luxury")[1] is False      # not in enum
    assert coerce(FLIGHT.arg("cabin"), "BUSINESS")[0] == "business"  # case-normalized to enum


def _complete(origin="Dubai", dest="London", date="2026-09-25", **extra):
    d = {a.name: FieldDecision.resolved(a.name, v)
         for a, v in ((FLIGHT.arg("origin"), origin), (FLIGHT.arg("destination"), dest), (FLIGHT.arg("date"), date))}
    d.update({k: FieldDecision.resolved(k, v) for k, v in extra.items()})
    return d


def test_act_only_when_all_required_resolved_and_valid():
    c = compile_call(FLIGHT, _complete(passengers=2, cabin="business"))
    assert c.outcome == "ACT"
    assert c.args == {"origin": "Dubai", "destination": "London",
                      "date": "2026-09-25", "passengers": 2, "cabin": "business"}
    ok, errs = is_schema_valid(FLIGHT, c.args)
    assert ok, errs


def test_missing_required_becomes_ask_for_info_never_invented():
    d = {a.name: FieldDecision.resolved(a.name, "London")
         for a in (FLIGHT.arg("destination"),)}
    c = compile_call(FLIGHT, d)
    assert c.outcome == "ASK_FOR_INFO"
    # origin/date are absent from args entirely - the compiler did NOT fill them
    assert "origin" not in c.args and "date" not in c.args
    assert set(c.missing) == {"origin", "date"}


def test_ambiguous_required_becomes_ask_for_info():
    d = _complete()
    d["origin"] = FieldDecision.ambiguous("origin", ["Dubai", "Abu Dhabi"])
    c = compile_call(FLIGHT, d)
    assert c.outcome == "ASK_FOR_INFO"
    assert "origin" in c.ambiguous
    assert "origin" not in c.args  # refuses to pick one


def test_invalid_value_does_not_satisfy_required_field():
    d = _complete(date="whenever")
    c = compile_call(FLIGHT, d)
    assert c.outcome == "ASK_FOR_INFO"
    assert "date" in c.missing  # failed validation -> treated as unresolved


def test_unsupported_arg_is_rejected_not_emitted():
    c = compile_call(FLIGHT, _complete(), extra_args=("seat",))
    assert c.outcome == "REJECT"
    assert any("unsupported" in r for r in c.reasons)


def test_nested_object_shape_comes_from_schema():
    card = ToolCard(
        tool_id="add_stop", name="add_stop", description="Add a stop to a trip",
        args=(ArgSpec("trip", "string", True, "trip id"),
              ArgSpec("stop", "object", True, "stop details",
                      fields=(ArgSpec("city", "string", True, "city"),
                              ArgSpec("nights", "integer", False, "nights"))),
              ArgSpec("legs", "array", False, "legs", item_type="string")))
    d = {"trip": FieldDecision.resolved("trip", "T-1"),
         "stop": FieldDecision.resolved("stop", {"city": "Rome", "nights": 2})}
    c = compile_call(card, d)
    assert c.outcome == "ACT"
    assert c.args["stop"] == {"city": "Rome", "nights": 2}
    # nested missing required -> ask, not invent
    d2 = {"trip": FieldDecision.resolved("trip", "T-1"),
          "stop": FieldDecision.resolved("stop", {"nights": 2})}
    c2 = compile_call(card, d2)
    assert c2.outcome == "ASK_FOR_INFO"
    assert "stop.city" in c2.missing


def test_schema_validity_is_independent_of_field_order():
    a = {"origin": "Dubai", "destination": "London", "date": "2026-09-25"}
    b = {"date": "2026-09-25", "destination": "London", "origin": "Dubai"}
    ok1, _ = is_schema_valid(FLIGHT, a)
    ok2, _ = is_schema_valid(FLIGHT, b)
    assert ok1 == ok2 is True


def test_extract_slot_never_invents_a_free_text_value():
    d = extract_slot(FLIGHT.arg("origin"), "Book me a flight to London.")
    assert d.state == MISSING  # refuses to guess the city
    ok, _ = coerce(FLIGHT.arg("origin"), d.value)
    assert not ok


def test_extract_slot_resolves_typed_values_exactly():
    st = "Fly Dubai to London on 2026-09-25, business class, 2 passengers."
    assert extract_slot(FLIGHT.arg("date"), st).value == "2026-09-25"
    assert extract_slot(FLIGHT.arg("passengers"), st).value == 2
    assert extract_slot(FLIGHT.arg("cabin"), st).value == "business"
