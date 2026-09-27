from scripts.eval.analyze_prem_deltas import main, option_jaccard


def test_option_jaccard():
    assert option_jaccard(["uses a mop", "uses a cloth"]) == 0.5


def test_delta_report(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.jsonl"
    memory = tmp_path / "memory.jsonl"
    baseline.write_text(
        '{"id": "a", "answer": 0, "pred": "A", "question": "(A) mop\\n(B) cloth"}\n'
        '{"id": "b", "answer": 1, "pred": "A", "question": "(A) mop\\n(B) cloth"}\n',
        encoding="utf-8",
    )
    memory.write_text(
        '{"id": "a", "answer": 0, "pred": "A", "question": "(A) mop\\n(B) cloth"}\n'
        '{"id": "b", "answer": 1, "pred": "B", "question": "(A) mop\\n(B) cloth"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_prem_deltas.py", "--baseline", str(baseline), "--memory", str(memory), "--output", str(output)],
    )
    main()
    report = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["harmed"] == 0
    assert report["summary"]["rescued"] == 1
