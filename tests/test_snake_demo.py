"""Regression tests for Snake's decision and recording contracts."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


import examples.snake.snake as snake_runtime
from examples.snake.snake import SnakeGame, Trust, move, replay, run
from examples.snake.snake_tui import DIGITS, DISPLAY_DIGITS, GREEN, Grid


class FixedScorer:
    def score_rows(self, rows):
        return [0.99, 0.01, 0.01, 0.01]


def test_raw_head_can_execute_illegal_move_while_shield_blocks_it():
    raw_game = SnakeGame(seed=7)
    raw_game.snake = [(3, 0), (3, 1), (3, 2)]
    raw_game.direction = "RIGHT"
    raw_game.food = (5, 5)
    raw = move(raw_game, Trust(), FixedScorer(), shield=False)
    assert raw["raw_proposal"] == "UP"
    assert raw["illegal_raw"] and not raw["after"]["alive"]

    shield_game = SnakeGame(seed=7)
    shield_game.snake = [(3, 0), (3, 1), (3, 2)]
    shield_game.direction = "RIGHT"
    shield_game.food = (5, 5)
    guarded = move(shield_game, Trust(), FixedScorer(), shield=True)
    assert guarded["after"]["alive"]
    assert guarded["executed"] != "UP"


def test_replay_rejects_changed_transition_and_gate(tmp_path):
    args = SimpleNamespace(width=24, height=16, seed=3, fps=12, max_speed=True,
        headless=True, steps=8, duration=None, trust_floor=.5, record=str(tmp_path / "moves.jsonl"),
        train_episodes=1, train_epochs=1, unassisted=False)
    run(args, scorer=FixedScorer(), backend="fixed-test")
    assert replay(args.record)["verified_moves"] == 8
    rows = [json.loads(line) for line in Path(args.record).read_text().splitlines()]
    rows[1]["gate"] = "MODEL" if rows[1]["gate"] != "MODEL" else "SAFETY SHIELD"
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="gate decision mismatch"):
        replay(tampered)
    rows[1]["gate"] = json.loads(Path(args.record).read_text().splitlines()[1])["gate"]
    rows[1]["after"]["score"] += 1
    tampered.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="transition mismatch"):
        replay(tampered)


def test_board_growth_and_tail_vacancy_are_replayed_as_real_transitions():
    game = SnakeGame(seed=5)
    game.snake = [(4, 4), (3, 4), (3, 5)]
    game.food = (5, 4)
    assert game.step("RIGHT")
    assert game.snake == [(5, 4), (4, 4), (3, 4), (3, 5)]
    assert game.score == 1
    game.food = (0, 0)
    assert game.step("DOWN")
    assert game.snake == [(5, 5), (5, 4), (4, 4), (3, 4)]


def test_benchmark_keeps_failed_per_seed_deadlines_and_resumes(monkeypatch, tmp_path):
    monkeypatch.setattr(snake_runtime, "prepare", lambda args: (FixedScorer(), "fixed-test"))
    args = SimpleNamespace(width=24, height=16, seed=1, trust_floor=.5, unassisted=True,
        seeds="101,102", rates="100000", steps=2, sweep_steps=2, soak_steps=2,
        output=str(tmp_path / "report.json"), train_episodes=1, train_epochs=1, resume=False)
    report = snake_runtime.benchmark(args)
    assert len(report["uncapped"]) == 2
    assert len(report["sweep"]["100000"]["seeds"]) == 2
    assert not report["sweep"]["100000"]["pass"]
    assert report["soak"] == {}
    assert report["highest_sustained_fps"] is None
    assert report["uncapped_aggregate"]["total_steps"] == 4
    class NoScoring:
        def score_rows(self, rows):
            raise AssertionError("resume recomputed a completed seed")
    monkeypatch.setattr(snake_runtime, "prepare", lambda args: (NoScoring(), "fixed-test"))
    args.resume = True
    resumed = snake_runtime.benchmark(args)
    assert resumed["uncapped"] == report["uncapped"]
    assert resumed["sweep"] == report["sweep"]


def test_large_digit_bitmaps_have_fixed_single_cell_geometry():
    assert set(DIGITS) == set("0123456789")
    assert all(len(glyph) == 5 for glyph in DIGITS.values())
    assert all(len(row) == 3 and set(row) <= {"█", " "}
               for glyph in DIGITS.values() for row in glyph)
    assert all(len(glyph) == 3 and all(len(row) == 3 and set(row) <= {"█", "▀", "▄", " "}
               for row in glyph) for glyph in DISPLAY_DIGITS.values())


@pytest.mark.parametrize("value", [0, 1, 2, 12, 34, 111, 222, 888, 999])
def test_large_number_blits_three_fixed_width_digits_with_leading_zeroes(value):
    grid = Grid(12, 3)
    grid.big_number(0, 0, value, GREEN)
    for row in range(3):
        expected = " ".join(DISPLAY_DIGITS[digit][row] for digit in f"{value:03d}")
        assert "".join(grid.cells[row][:11]) == expected
        assert all(grid.colors[row][column] == GREEN
                   for column, pixel in enumerate(expected) if pixel != " ")


@pytest.mark.parametrize("score,length,best", [(2, 8, 2), (34, 40, 34), (111, 222, 888), (999, 100, 1)])
def test_metric_groups_are_centered_in_separate_sixteen_cell_columns(score, length, best):
    grid = Grid(48, 3)
    for start, value in zip((2, 18, 34), (score, length, best)):
        grid.big_number(0, start, value, GREEN)
    for row in range(3):
        for start, value in zip((2, 18, 34), (score, length, best)):
            expected = " ".join(DISPLAY_DIGITS[digit][row] for digit in f"{value:03d}")
            assert "".join(grid.cells[row][start:start + 11]) == expected
        assert "".join(grid.cells[row][13:18]) == " " * 5
        assert "".join(grid.cells[row][29:34]) == " " * 5
