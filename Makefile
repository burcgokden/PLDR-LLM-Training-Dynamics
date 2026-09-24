.PHONY: check test lean
check:
	python3 scripts/verify_scientific_manifest.py
	python3 scripts/check_formal_manifest.py
test:
	python3 scripts/run_scientific_checks.py
lean:
	lake build
	python3 scripts/check_lean.py
