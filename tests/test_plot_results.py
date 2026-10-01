import json
from pathlib import Path

from scripts.plot_results import create_plots, load_report


def test_plot_script_reads_evaluator_json_and_generates_all_figures(tmp_path: Path):
    controllers = ("Random", "Rule-based", "DQN", "PPO")
    rows = []
    for scenario in range(2):
        for offset, controller in enumerate(controllers):
            rows.append(
                {
                    "scenario_index": scenario,
                    "controller": controller,
                    "reward": float(offset + scenario),
                    "total_waiting": float(10 - offset + scenario),
                    "max_queue": 8 - offset + scenario,
                    "vehicles_passed": 20 + offset + scenario,
                    "violations": 4 - offset,
                }
            )
    report = {
        "scenarios": [{"scenario_index": 0}, {"scenario_index": 1}],
        "episode_results": rows,
        "summaries": [],
        "ppo_minus_dqn": [
            {"metric": metric, "mean": 1.0, "std": 0.2, "ci95_low": 0.8, "ci95_high": 1.2}
            for metric in ("reward", "waiting", "vehicles_passed", "violations")
        ],
    }
    input_path = tmp_path / "evaluation.json"
    input_path.write_text(json.dumps(report))
    loaded = load_report(input_path)
    generated = create_plots(loaded, tmp_path / "plots")
    assert len(generated) == 18
    assert all(path.stat().st_size > 0 for path in generated)
