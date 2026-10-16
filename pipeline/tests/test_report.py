from accuscan.report import render_report

METRICS = {
    "sets": {
        "gnhk_test_words": {
            "greedy": {"cer": 0.2, "wer": 0.4, "samples": 10},
            "dictionary": {"cer": 0.15, "wer": 0.3, "samples": 10},
        },
        "canvas_heldout_fonts": {"greedy": {"cer": 0.1, "wer": 0.2, "samples": 5}},
    },
    "gate": {"canvas_cer": 0.1, "threshold": 0.15, "defaultMode": "free"},
    "latency_cpu": {"model_mb": 4.8, "ms_width_256": 12.5},
}


def test_report_without_metrics_says_so() -> None:
    assert "No trained model" in render_report(None, None)


def test_report_tables_come_from_the_json() -> None:
    text = render_report(METRICS, None)
    assert "| GNHK test, words | 10 | 20.0% | 40.0% | 15.0% | 30.0% |" in text
    assert "opens in **free** mode" in text
    assert "12.5 ms at width 256" in text


def test_report_adds_the_baseline_comparison() -> None:
    baseline = {
        "model": "microsoft/trocr-small-handwritten",
        "model_mb": 246.0,
        "sets": {"gnhk_test_words": {"cer": 0.1, "wer": 0.2, "samples": 3, "ms_per_crop": 300.0}},
    }
    text = render_report(METRICS, baseline)
    assert "| microsoft/trocr-small-handwritten | GNHK test, words | 3 | 10.0% |" in text
