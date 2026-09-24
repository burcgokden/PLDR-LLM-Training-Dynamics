from __future__ import annotations

from pathlib import Path
import sys


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from qualification_normal_stability import qualify  # noqa: E402


def test_live_q0_owns_target_rows_and_exact_constructors():
    result = qualify()
    assert result["decision"] == "QUALIFIED"
    assert all(result["conditions"].values())
    diagnostics = result["jvp_and_taylor"]
    assert diagnostics["physical_row_displacement"] > 0
    assert diagnostics["moving_row_decomposition_error"] < 1e-12
    assert diagnostics["dense_action_error"] < 1e-12
    assert diagnostics["remainder_enclosed"]
    assert diagnostics["row_count"] > 1
    assert diagnostics["energy_identity_error"] < 1e-12
    assert result["replay"]["decision"] == "REPLAYED"
    assert result["resource_metadata"]["model_state_bytes"] > 0
    assert result["resource_metadata"]["optimizer_state_bytes"] > 0
