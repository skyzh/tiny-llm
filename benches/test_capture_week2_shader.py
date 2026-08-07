from types import SimpleNamespace

from benches import capture_week2_shader as capture_module


def test_capture_retires_pre_capture_buffer_before_measured_evals(
    monkeypatch, tmp_path
):
    events = []
    output = tmp_path / "week2.gputrace"
    args = SimpleNamespace(output=output, iterations=2)

    def build(name):
        events.append(f"build:{name}")
        return [name]

    fake_metal = SimpleNamespace(
        start_capture=lambda path: events.append("start"),
        stop_capture=lambda: events.append("stop"),
    )
    fake_mx = SimpleNamespace(
        metal=fake_metal,
        eval=lambda value: events.append(f"eval:{value}"),
        synchronize=lambda: events.append("sync"),
    )

    monkeypatch.setattr(capture_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        capture_module,
        "prepare_workload",
        lambda parsed: (
            "test workload",
            lambda: build("warmup"),
            lambda: build("capture"),
        ),
    )
    monkeypatch.setattr(capture_module, "mx", fake_mx)

    capture_module.main()

    assert events == [
        "build:warmup",
        "eval:warmup",
        "build:capture",
        "eval:capture",
        "start",
        "sync",
        "build:capture",
        "eval:capture",
        "build:capture",
        "eval:capture",
        "sync",
        "stop",
    ]
