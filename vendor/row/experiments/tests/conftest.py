import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------
# Relocatable manuscript tree: the three checks that read rendered
# manuscript artifacts resolve their root through ONE variable, so a
# public package passes without intervention: set
# PLDR_MANUSCRIPT_DIR to the directory holding
# pldr_training_dynamics.tex (with its figures/ subdirectory), or
# rely on the in-repo default docs/.  When the tree is absent the
# dependent tests skip with an instruction rather than fail on a
# packaging path.
import pytest  # noqa: E402


def manuscript_root():
    env = os.environ.get("PLDR_MANUSCRIPT_DIR")
    if env:
        return os.path.abspath(env)
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "docs")


def require_manuscript_root():
    root = manuscript_root()
    if not os.path.isdir(root):
        pytest.skip(
            "manuscript tree not present at %r; set "
            "PLDR_MANUSCRIPT_DIR to the directory holding "
            "pldr_training_dynamics.tex" % root)
    return root
