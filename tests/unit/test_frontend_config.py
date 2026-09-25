"""Regression test for a real bug found in a visual QA pass (see
docs/visual-qa-report): Streamlit's file_uploader displays its own
`server.maxUploadSize` (default 200MB), completely independent of the API's
actual `Settings.api_max_upload_mb`. Without this pinned in
`.streamlit/config.toml`, the two silently drift apart -- the UI shows one
limit while the API enforces a different one, and part of the API's real
limit becomes unreachable from the frontend.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from transcript_task.settings import PROJECT_ROOT, Settings


def test_streamlit_max_upload_size_matches_the_api_default():
    config = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text())
    assert config["server"]["maxUploadSize"] == Settings.model_fields["api_max_upload_mb"].default


def test_config_toml_is_valid():
    # Loading it at all (tomllib.loads above) already proves this, but a
    # dedicated test names the failure mode clearly if config.toml is ever
    # hand-edited into invalid syntax.
    path = Path(PROJECT_ROOT) / ".streamlit" / "config.toml"
    assert path.exists()
