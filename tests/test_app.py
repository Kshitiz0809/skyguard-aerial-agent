"""Headless UI test of the Streamlit console using streamlit.testing."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_launch_pause_and_resume(samples_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.sidebar.button[0].click().run()  # Launch mission (default scenario: substation_intruder)
    assert not at.exception
    assert any("Human approval required" in s.value for s in at.subheader)

    submit = next(b for b in at.button if "Submit decisions" in b.label)
    submit.click().run()
    assert not at.exception
    assert not any("Human approval required" in s.value for s in at.subheader)
    assert any("MISSION COMPLETE" in m.value for m in at.markdown)
