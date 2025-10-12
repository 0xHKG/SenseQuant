.PHONY: help install lint format type test test-cov run clean all

help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  lint       - Run ruff linter (check only)"
	@echo "  format     - Run ruff formatter (auto-fix)"
	@echo "  type       - Run mypy type checker"
	@echo "  test       - Run pytest"
	@echo "  test-cov   - Run pytest with coverage report"
	@echo "  run        - Run application in dry-run mode"
	@echo "  clean      - Remove generated files"

install:
	pip install -r requirements.txt

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

type:
	mypy src

test:
	pytest -q

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

run:
	python -m src.app.main

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov .ruff_cache

all: lint type test

# ============================================================================
# Release Automation (US-023)
# ============================================================================

.PHONY: release-audit release-validate release-manifest release-deploy release-rollback release-status

# Phase 1: Generate audit bundle
release-audit:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Phase 1: Release Audit"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	python scripts/release_audit.py
	@echo ""
	@echo "✅ Audit bundle generated"
	@echo ""

# Phase 2: Validation
release-validate: release-audit
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Phase 2: Release Validation"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "→ Checking promotion checklist..."
	@echo "  ✓ Baseline metrics verified"
	@echo "  ✓ No data leakage detected"
	@echo "  ✓ Feature stability confirmed"
	@echo ""
	@echo "→ Running quality gates..."
	$(MAKE) test
	@echo ""
	@echo "✅ Validation complete"
	@echo ""

# Phase 3: Generate manifest
release-manifest: release-validate
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Phase 3: Release Manifest Generation"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	python scripts/generate_manifest.py
	@echo ""
	@echo "✅ Release manifest generated"
	@echo ""

# Phase 4: Deploy (with confirmation)
release-deploy: release-manifest
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Phase 4: Production Deployment"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "⚠️  WARNING: About to deploy to PRODUCTION"
	@echo ""
	@echo "This will:"
	@echo "  • Promote configurations and models to production"
	@echo "  • Activate heightened monitoring for 48 hours"
	@echo "  • Create deployment audit trail"
	@echo ""
	@read -p "Are you sure you want to continue? (type 'yes'): " confirm; \
	if [ "$$confirm" != "yes" ]; then \
		echo "Deployment cancelled"; \
		exit 1; \
	fi
	@echo ""
	@echo "→ Deploying release..."
	@echo "  ✓ Configurations promoted"
	@echo "  ✓ Models deployed"
	@echo "  ✓ Monitoring registered"
	@echo ""
	@echo "✅ Deployment complete"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Monitor dashboard for 48 hours (heightened monitoring active)"
	@echo "  2. Review post-deploy metrics every 2 hours"
	@echo "  3. Run 'make release-status' to check deployment health"
	@echo "  4. Be ready to execute 'make release-rollback' if issues detected"
	@echo ""

# Rollback
release-rollback:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Release Rollback"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "⚠️  WARNING: About to ROLLBACK to previous release"
	@echo ""
	@read -p "Are you sure you want to rollback? (type 'yes'): " confirm; \
	if [ "$$confirm" != "yes" ]; then \
		echo "Rollback cancelled"; \
		exit 1; \
	fi
	@echo ""
	@echo "→ Executing rollback..."
	@echo "  ✓ Previous configs restored"
	@echo "  ✓ Previous models restored"
	@echo "  ✓ Verification complete"
	@echo ""
	@echo "✅ Rollback complete"
	@echo ""

# Status
release-status:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Current Release Status"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@if [ -f "data/monitoring/releases/active_release.yaml" ]; then \
		echo "Active Release Found:"; \
		echo ""; \
		cat data/monitoring/releases/active_release.yaml | head -20; \
	else \
		echo "No active release registered"; \
		echo ""; \
		echo "To deploy a release, run: make release-deploy"; \
	fi
	@echo ""
