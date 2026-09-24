.PHONY: lean-test formal-audit
lean-test:
	lake build
formal-audit: lean-test
	python3 scripts/check_formal_soundness.py
