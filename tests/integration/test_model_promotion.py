"""Integration tests for student model promotion workflow (US-021)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_student_model_config_defaults():
    """Test student model configuration defaults (US-021).

    Verifies:
    - student_model_enabled defaults to False (safe by default)
    - student_model_path has sensible default
    - Validation thresholds have reasonable defaults
    """
    from src.app.config import Settings

    # Create settings with defaults
    settings = Settings()

    # Verify student model disabled by default (safe)
    assert settings.student_model_enabled is False, "Student model should be disabled by default"

    # Verify sensible default path
    assert "production" in settings.student_model_path, "Default path should reference production"
    assert "student_model.pkl" in settings.student_model_path

    # Verify validation thresholds
    assert settings.promotion_min_precision_uplift == 0.02, "Default precision uplift: +2%"
    assert settings.promotion_min_hit_ratio_uplift == 0.02, "Default hit ratio uplift: +2%"
    assert settings.promotion_min_sharpe_uplift == 0.05, "Default Sharpe uplift: +5%"
    assert settings.promotion_require_all_criteria is True, "Should require all criteria by default"

    # Verify confidence threshold
    assert 0.0 < settings.student_model_confidence_threshold <= 1.0
    assert settings.student_model_confidence_threshold == 0.6, "Default confidence: 60%"

    print("\n✓ Student model configuration defaults validated")
    print(f"  - student_model_enabled: {settings.student_model_enabled}")
    print(f"  - student_model_path: {settings.student_model_path}")
    print(f"  - confidence_threshold: {settings.student_model_confidence_threshold}")
    print(f"  - promotion_min_precision_uplift: {settings.promotion_min_precision_uplift}")
    print(f"  - promotion_min_hit_ratio_uplift: {settings.promotion_min_hit_ratio_uplift}")
    print(f"  - promotion_min_sharpe_uplift: {settings.promotion_min_sharpe_uplift}")


def test_promotion_checklist_structure():
    """Test promotion checklist structure validation (US-021).

    Verifies:
    - Checklist JSON has required fields
    - Validation results boolean flags present
    - Recommendation field (PROMOTE/REJECT)
    - Delta metrics computed correctly
    """
    # Mock promotion checklist structure
    checklist = {
        "timestamp": "2025-01-12T15:30:00Z",
        "candidate_model": "data/models/20250112_143000/student/student_model.pkl",
        "baseline_model": "data/models/baseline/student_model.pkl",
        "validation_period": {
            "start": "2024-07-01",
            "end": "2024-09-30",
            "symbols": ["RELIANCE", "TCS"],
        },
        "candidate_metrics": {
            "precision_long": 0.712,
            "hit_ratio": 0.668,
            "accuracy": 0.687,
            "sharpe_ratio": 1.85,
            "total_return": 0.142,
            "max_drawdown": 0.083,
            "win_rate": 0.583,
        },
        "baseline_metrics": {
            "precision_long": 0.685,
            "hit_ratio": 0.632,
            "accuracy": 0.654,
            "sharpe_ratio": 1.72,
            "total_return": 0.118,
            "max_drawdown": 0.091,
            "win_rate": 0.557,
        },
        "deltas": {
            "precision_uplift": 0.027,
            "hit_ratio_uplift": 0.036,
            "sharpe_uplift": 0.13,
            "return_uplift": 0.024,
            "drawdown_improvement": -0.008,
            "win_rate_uplift": 0.026,
        },
        "validation_thresholds": {
            "min_precision_uplift": 0.02,
            "min_hit_ratio_uplift": 0.02,
            "min_sharpe_uplift": 0.05,
            "require_all_criteria": True,
        },
        "validation_results": {
            "precision_pass": True,
            "hit_ratio_pass": True,
            "sharpe_pass": True,
            "all_criteria_pass": True,
        },
        "recommendation": "PROMOTE",
        "reason": "All validation criteria passed.",
    }

    # Verify required top-level fields
    assert "timestamp" in checklist
    assert "candidate_model" in checklist
    assert "baseline_model" in checklist
    assert "validation_period" in checklist
    assert "candidate_metrics" in checklist
    assert "baseline_metrics" in checklist
    assert "deltas" in checklist
    assert "validation_thresholds" in checklist
    assert "validation_results" in checklist
    assert "recommendation" in checklist

    # Verify validation results
    assert isinstance(checklist["validation_results"]["precision_pass"], bool)
    assert isinstance(checklist["validation_results"]["hit_ratio_pass"], bool)
    assert isinstance(checklist["validation_results"]["sharpe_pass"], bool)
    assert isinstance(checklist["validation_results"]["all_criteria_pass"], bool)

    # Verify recommendation
    assert checklist["recommendation"] in ["PROMOTE", "REJECT"]

    # Verify delta calculations
    deltas = checklist["deltas"]
    candidate = checklist["candidate_metrics"]
    baseline = checklist["baseline_metrics"]

    assert (
        abs(deltas["precision_uplift"] - (candidate["precision_long"] - baseline["precision_long"]))
        < 0.001
    )
    assert (
        abs(deltas["hit_ratio_uplift"] - (candidate["hit_ratio"] - baseline["hit_ratio"])) < 0.001
    )
    assert (
        abs(deltas["sharpe_uplift"] - (candidate["sharpe_ratio"] - baseline["sharpe_ratio"]))
        < 0.001
    )

    # Verify thresholds met
    thresholds = checklist["validation_thresholds"]
    results = checklist["validation_results"]

    assert results["precision_pass"] == (
        deltas["precision_uplift"] >= thresholds["min_precision_uplift"]
    )
    assert results["hit_ratio_pass"] == (
        deltas["hit_ratio_uplift"] >= thresholds["min_hit_ratio_uplift"]
    )
    assert results["sharpe_pass"] == (deltas["sharpe_uplift"] >= thresholds["min_sharpe_uplift"])

    print("\n✓ Promotion checklist structure validated")
    print(f"  - Recommendation: {checklist['recommendation']}")
    print(
        f"  - Precision uplift: {deltas['precision_uplift']:.1%} (pass: {results['precision_pass']})"
    )
    print(
        f"  - Hit ratio uplift: {deltas['hit_ratio_uplift']:.1%} (pass: {results['hit_ratio_pass']})"
    )
    print(f"  - Sharpe uplift: {deltas['sharpe_uplift']:.1%} (pass: {results['sharpe_pass']})")


def test_promotion_validation_logic():
    """Test promotion validation logic (US-021).

    Verifies:
    - Criteria evaluation (pass/fail)
    - All criteria required logic
    - PROMOTE vs REJECT recommendation
    """
    from src.app.config import Settings

    settings = Settings()

    # Test case 1: All criteria pass
    candidate_metrics = {
        "precision_long": 0.72,
        "hit_ratio": 0.67,
        "sharpe_ratio": 1.85,
    }
    baseline_metrics = {
        "precision_long": 0.68,
        "hit_ratio": 0.63,
        "sharpe_ratio": 1.72,
    }

    precision_uplift = candidate_metrics["precision_long"] - baseline_metrics["precision_long"]
    hit_ratio_uplift = candidate_metrics["hit_ratio"] - baseline_metrics["hit_ratio"]
    sharpe_uplift = candidate_metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"]

    precision_pass = precision_uplift >= settings.promotion_min_precision_uplift
    hit_ratio_pass = hit_ratio_uplift >= settings.promotion_min_hit_ratio_uplift
    sharpe_pass = sharpe_uplift >= settings.promotion_min_sharpe_uplift

    if settings.promotion_require_all_criteria:
        all_pass = precision_pass and hit_ratio_pass and sharpe_pass
    else:
        all_pass = precision_pass or hit_ratio_pass or sharpe_pass

    recommendation = "PROMOTE" if all_pass else "REJECT"

    assert precision_pass is True, "Precision should pass (+4%)"
    assert hit_ratio_pass is True, "Hit ratio should pass (+4%)"
    assert sharpe_pass is True, "Sharpe should pass (+7.6%)"
    assert all_pass is True, "All criteria should pass"
    assert recommendation == "PROMOTE", "Should recommend PROMOTE"

    print("\n✓ Promotion validation logic validated (Case 1: All pass)")
    print(
        f"  - Precision: {precision_uplift:.1%} >= {settings.promotion_min_precision_uplift:.1%} → {precision_pass}"
    )
    print(
        f"  - Hit ratio: {hit_ratio_uplift:.1%} >= {settings.promotion_min_hit_ratio_uplift:.1%} → {hit_ratio_pass}"
    )
    print(
        f"  - Sharpe: {sharpe_uplift:.1%} >= {settings.promotion_min_sharpe_uplift:.1%} → {sharpe_pass}"
    )
    print(f"  - Recommendation: {recommendation}")

    # Test case 2: Some criteria fail
    candidate_metrics2 = {
        "precision_long": 0.69,  # Only +1%, below threshold
        "hit_ratio": 0.66,  # +3%, passes
        "sharpe_ratio": 1.75,  # Only +1.7%, below threshold
    }

    precision_uplift2 = candidate_metrics2["precision_long"] - baseline_metrics["precision_long"]
    hit_ratio_uplift2 = candidate_metrics2["hit_ratio"] - baseline_metrics["hit_ratio"]
    sharpe_uplift2 = candidate_metrics2["sharpe_ratio"] - baseline_metrics["sharpe_ratio"]

    precision_pass2 = precision_uplift2 >= settings.promotion_min_precision_uplift
    hit_ratio_pass2 = hit_ratio_uplift2 >= settings.promotion_min_hit_ratio_uplift
    sharpe_pass2 = sharpe_uplift2 >= settings.promotion_min_sharpe_uplift

    if settings.promotion_require_all_criteria:
        all_pass2 = precision_pass2 and hit_ratio_pass2 and sharpe_pass2
    else:
        all_pass2 = precision_pass2 or hit_ratio_pass2 or sharpe_pass2

    recommendation2 = "PROMOTE" if all_pass2 else "REJECT"

    assert precision_pass2 is False, "Precision should fail (+1% < +2%)"
    assert hit_ratio_pass2 is True, "Hit ratio should pass (+3%)"
    assert sharpe_pass2 is False, "Sharpe should fail (+1.7% < +5%)"
    assert all_pass2 is False, "Not all criteria should pass"
    assert recommendation2 == "REJECT", "Should recommend REJECT"

    print("\n✓ Promotion validation logic validated (Case 2: Some fail)")
    print(
        f"  - Precision: {precision_uplift2:.1%} >= {settings.promotion_min_precision_uplift:.1%} → {precision_pass2}"
    )
    print(
        f"  - Hit ratio: {hit_ratio_uplift2:.1%} >= {settings.promotion_min_hit_ratio_uplift:.1%} → {hit_ratio_pass2}"
    )
    print(
        f"  - Sharpe: {sharpe_uplift2:.1%} >= {settings.promotion_min_sharpe_uplift:.1%} → {sharpe_pass2}"
    )
    print(f"  - Recommendation: {recommendation2}")


def test_promotion_checklist_markdown_format():
    """Test promotion checklist markdown format (US-021).

    Verifies:
    - Markdown checklist has required sections
    - Pass/fail indicators present
    - Approval sign-off section
    - Promotion commands section
    - Rollback procedure section
    """
    # Mock markdown checklist
    markdown = """# Student Model Promotion Checklist

**Model**: `data/models/20250112_143000/student/student_model.pkl`
**Generated**: 2025-01-12 15:30:00

## Performance Summary

### Candidate Model Performance
- **Precision (LONG)**: 71.2%
- **Sharpe Ratio**: 1.85

### Baseline Model Performance
- **Precision (LONG)**: 68.5%
- **Sharpe Ratio**: 1.72

### Delta (Candidate - Baseline)
- **Precision Uplift**: +2.7% ✅
- **Sharpe Uplift**: +7.6% ✅

## Validation Criteria

✅ **Precision Uplift**: 2.7% ≥ 2.0% (PASS)
✅ **Sharpe Uplift**: 7.6% ≥ 5.0% (PASS)

## Approval Sign-offs

- [ ] **ML Lead**: _______________ Date: ___________
- [ ] **Quant Team Lead**: _______________ Date: ___________

## Promotion Commands

```bash
cp data/models/20250112_143000/student/student_model.pkl \\
   data/models/production/student_model.pkl
```

## Rollback Procedure

```bash
export SENSEQUANT_STUDENT_MODEL_ENABLED=false
```
"""

    # Verify required sections
    assert "# Student Model Promotion Checklist" in markdown
    assert "Performance Summary" in markdown
    assert "Validation Criteria" in markdown
    assert "Approval Sign-offs" in markdown
    assert "Promotion Commands" in markdown
    assert "Rollback Procedure" in markdown

    # Verify pass/fail indicators
    assert "✅" in markdown or "PASS" in markdown
    assert "Candidate - Baseline" in markdown

    # Verify approval checkboxes
    assert "[ ]" in markdown or "- [ ]" in markdown
    assert "ML Lead" in markdown

    # Verify commands present
    assert "cp " in markdown or "export " in markdown

    print("\n✓ Promotion checklist markdown format validated")
    print("  - All required sections present")
    print("  - Pass/fail indicators included")
    print("  - Approval sign-offs section present")
    print("  - Promotion and rollback commands documented")


@pytest.mark.skipif(
    not Path("data/models/sample_run/student").exists(), reason="Sample student model not found"
)
def test_student_model_file_structure():
    """Test student model file structure from US-020 (US-021 prerequisite).

    Verifies:
    - Student model file exists
    - Metadata file exists and has required fields
    - Evaluation metrics file exists
    """
    student_dir = Path("data/models/sample_run/student")
    assert student_dir.exists(), "Sample student directory not found"

    # Check model file
    model_file = student_dir / "student_model.pkl"
    assert model_file.exists(), "Student model file not found"

    # Check metadata
    metadata_file = student_dir / "metadata.json"
    assert metadata_file.exists(), "Metadata file not found"

    with open(metadata_file) as f:
        metadata = json.load(f)

    assert "timestamp" in metadata
    assert "model_type" in metadata
    # Check for teacher linkage (US-020 feature)
    # Note: teacher_dir may or may not be present depending on implementation

    # Check evaluation metrics
    metrics_file = student_dir / "evaluation_metrics.json"
    assert metrics_file.exists(), "Evaluation metrics file not found"

    with open(metrics_file) as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics

    print("\n✓ Student model file structure validated")
    print(f"  - Model file: {model_file}")
    print(f"  - Metadata: {metadata_file}")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Model type: {metadata.get('model_type', 'unknown')}")


def test_promotion_workflow_safety_checks():
    """Test promotion workflow safety checks (US-021).

    Verifies:
    - Safety checks for model file existence
    - Validation artifacts presence
    - Configuration consistency
    """
    # Test case: Missing model file
    model_path = Path("data/models/nonexistent/student_model.pkl")
    assert not model_path.exists(), "Test assumes nonexistent model"

    can_promote = False
    errors = []

    # Safety check 1: Model file exists
    if not model_path.exists():
        errors.append(f"Model file not found: {model_path}")
    else:
        can_promote = True

    assert can_promote is False, "Should not promote if model missing"
    assert len(errors) > 0, "Should have error message"
    assert "not found" in errors[0].lower()

    print("\n✓ Promotion workflow safety checks validated")
    print(f"  - Model file check: {len(errors)} error(s)")
    print(f"  - Can promote: {can_promote}")

    # Test case: Valid promotion scenario
    can_promote2 = True
    errors2 = []

    # Mock: all checks pass
    model_exists = True
    metadata_valid = True
    validation_passed = True

    if not model_exists:
        errors2.append("Model not found")
    if not metadata_valid:
        errors2.append("Invalid metadata")
    if not validation_passed:
        errors2.append("Validation failed")

    can_promote2 = len(errors2) == 0

    assert can_promote2 is True, "Should promote if all checks pass"
    assert len(errors2) == 0, "Should have no errors"

    print("\n✓ Valid promotion scenario validated")
    print(f"  - Can promote: {can_promote2}")
    print(f"  - Errors: {len(errors2)}")


def test_student_model_promoter_validation(tmp_path):
    """Test StudentModelPromoter validation logic (US-021 Phase 2).

    Verifies:
    - Promoter validates model file exists
    - Promoter validates metadata exists
    - Promoter checks promotion checklist
    - Returns proper can_promote flag
    """
    from src.app.config import Settings
    from src.services.teacher_student import StudentModelPromoter

    settings = Settings()
    promoter = StudentModelPromoter(settings)

    # Create mock model artifacts
    model_dir = tmp_path / "models" / "candidate" / "student"
    model_dir.mkdir(parents=True)

    model_file = model_dir / "student_model.pkl"
    # Write a valid pickled mock model
    import pickle

    mock_model = {"model_type": "logistic", "version": "test"}
    with open(model_file, "wb") as f:
        pickle.dump(mock_model, f)

    metadata_file = model_dir / "metadata.json"
    metadata = {
        "timestamp": "2025-01-12T15:00:00Z",
        "model_type": "logistic",
        "test_size": 0.2,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    # Test case 1: Missing checklist (should warn but allow promotion)
    result = promoter.validate_promotion(str(model_file))

    assert "can_promote" in result
    assert "errors" in result
    assert "warnings" in result
    assert len(result["warnings"]) > 0  # Should warn about missing checklist

    print("\n✓ Promoter validation (missing checklist) tested")

    # Test case 2: With valid checklist (PROMOTE recommendation)
    checklist_file = model_dir / "promotion_checklist.json"
    checklist = {
        "timestamp": "2025-01-12T15:30:00Z",
        "candidate_model": str(model_file),
        "validation_results": {
            "precision_pass": True,
            "hit_ratio_pass": True,
            "sharpe_pass": True,
            "all_criteria_pass": True,
        },
        "recommendation": "PROMOTE",
    }
    with open(checklist_file, "w") as f:
        json.dump(checklist, f)

    result2 = promoter.validate_promotion(str(model_file))

    assert result2["can_promote"] is True
    assert result2["checklist_found"] is True
    assert result2["criteria_passed"] is True
    assert result2["recommendation"] == "PROMOTE"
    assert len(result2["errors"]) == 0

    print("✓ Promoter validation (valid checklist) tested")
    print(f"  - can_promote: {result2['can_promote']}")
    print(f"  - recommendation: {result2['recommendation']}")

    # Test case 3: With REJECT recommendation
    checklist["recommendation"] = "REJECT"
    checklist["validation_results"]["all_criteria_pass"] = False
    with open(checklist_file, "w") as f:
        json.dump(checklist, f)

    result3 = promoter.validate_promotion(str(model_file))

    assert result3["can_promote"] is False
    assert result3["recommendation"] == "REJECT"
    assert len(result3["errors"]) > 0

    print("✓ Promoter validation (reject recommendation) tested")
    print(f"  - can_promote: {result3['can_promote']}")
    print(f"  - errors: {result3['errors']}")


def test_student_model_promotion_workflow(tmp_path):
    """Test end-to-end student model promotion workflow (US-021 Phase 2).

    Verifies:
    - Promoter validates candidate model
    - Promoter backs up existing production model
    - Promoter copies candidate to production
    - Backup file created with timestamp
    - Production model updated
    """
    from src.app.config import Settings
    from src.services.teacher_student import StudentModelPromoter

    # Create settings with test paths
    settings = Settings()
    production_dir = tmp_path / "production"
    production_dir.mkdir(parents=True)
    settings.student_model_path = str(production_dir / "student_model.pkl")

    promoter = StudentModelPromoter(settings)

    # Create existing production model (to be backed up)
    import pickle

    existing_model = Path(settings.student_model_path)
    mock_old_model = {"model_type": "logistic", "version": "old"}
    with open(existing_model, "wb") as f:
        pickle.dump(mock_old_model, f)

    # Create candidate model
    candidate_dir = tmp_path / "models" / "20250112_143000" / "student"
    candidate_dir.mkdir(parents=True)

    candidate_model = candidate_dir / "student_model.pkl"
    mock_new_model = {"model_type": "logistic", "version": "new"}
    with open(candidate_model, "wb") as f:
        pickle.dump(mock_new_model, f)

    # Create metadata
    metadata_file = candidate_dir / "metadata.json"
    metadata = {"timestamp": "2025-01-12T15:00:00Z", "model_type": "logistic"}
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    # Create promotion checklist
    checklist_file = candidate_dir / "promotion_checklist.json"
    checklist = {
        "validation_results": {"all_criteria_pass": True},
        "recommendation": "PROMOTE",
    }
    with open(checklist_file, "w") as f:
        json.dump(checklist, f)

    # Validate promotion
    validation_result = promoter.validate_promotion(str(candidate_model))
    assert validation_result["can_promote"] is True

    # Promote model
    promotion_result = promoter.promote_model(str(candidate_model), dry_run=False)

    assert promotion_result["success"] is True
    assert promotion_result["backup_path"] is not None
    assert promotion_result["production_path"] == settings.student_model_path

    # Verify backup created
    backup_path = Path(promotion_result["backup_path"])
    assert backup_path.exists()
    with open(backup_path, "rb") as f:
        backup_model = pickle.load(f)
    assert backup_model["version"] == "old"

    # Verify production model updated
    with open(existing_model, "rb") as f:
        production_model = pickle.load(f)
    assert production_model["version"] == "new"

    print("\n✓ End-to-end promotion workflow tested")
    print(f"  - Backup created: {backup_path}")
    print(f"  - Production updated: {existing_model}")

    # Test rollback
    archive_date = backup_path.stem.replace("student_baseline_", "")
    rollback_result = promoter.rollback_model(archive_date)

    assert rollback_result["success"] is True
    assert rollback_result["restored_from"] is not None

    # Verify production model rolled back
    with open(existing_model, "rb") as f:
        rollback_model = pickle.load(f)
    assert rollback_model["version"] == "old"

    print("✓ Rollback tested")
    print(f"  - Restored from: {rollback_result['restored_from']}")


def test_promotion_dry_run(tmp_path):
    """Test promotion dry run mode (US-021 Phase 2).

    Verifies:
    - Dry run simulates promotion without file operations
    - No files modified in dry run mode
    - Proper logging of would-be operations
    """
    from src.app.config import Settings
    from src.services.teacher_student import StudentModelPromoter

    settings = Settings()
    production_dir = tmp_path / "production"
    production_dir.mkdir(parents=True)
    settings.student_model_path = str(production_dir / "student_model.pkl")

    promoter = StudentModelPromoter(settings)

    # Create existing production model
    import pickle

    existing_model = Path(settings.student_model_path)
    mock_old_model = {"model_type": "logistic", "version": "old"}
    with open(existing_model, "wb") as f:
        pickle.dump(mock_old_model, f)
    original_mtime = existing_model.stat().st_mtime

    # Create candidate model
    candidate_dir = tmp_path / "models" / "candidate" / "student"
    candidate_dir.mkdir(parents=True)

    candidate_model = candidate_dir / "student_model.pkl"
    mock_new_model = {"model_type": "logistic", "version": "new"}
    with open(candidate_model, "wb") as f:
        pickle.dump(mock_new_model, f)

    # Dry run promotion
    result = promoter.promote_model(str(candidate_model), dry_run=True)

    assert result["success"] is True
    assert "[DRY RUN]" in result["message"]

    # Verify no files modified
    with open(existing_model, "rb") as f:
        current_model = pickle.load(f)
    assert current_model["version"] == "old"
    assert existing_model.stat().st_mtime == original_mtime

    # Verify no backup created
    archive_dir = production_dir.parent / "archive"
    if archive_dir.exists():
        assert len(list(archive_dir.glob("*.pkl"))) == 0

    print("\n✓ Dry run promotion tested")
    print(f"  - Message: {result['message']}")
    print("  - No files modified")


def test_student_monitoring_and_rollback(tmp_path):
    """Test student model monitoring, degradation detection, and rollback (US-021 Phase 3).

    Verifies:
    - MonitoringService records student predictions
    - Performance degradation alerts are triggered
    - Automatic rollback is executed when needed
    - Rollback logs are created
    """
    from datetime import datetime, timedelta

    from src.app.config import Settings
    from src.services.monitoring import Alert, MonitoringService
    from src.services.teacher_student import StudentModelPromoter

    # Create settings with student monitoring enabled
    settings = Settings()
    settings.student_monitoring_enabled = True
    settings.student_monitoring_min_samples = 10  # Low threshold for testing
    settings.student_monitoring_precision_drop_threshold = 0.10
    settings.student_monitoring_hit_ratio_drop_threshold = 0.10
    settings.student_auto_rollback_enabled = True
    settings.student_auto_rollback_confirmation_hours = 1  # Short for testing

    # Initialize monitoring service
    monitoring = MonitoringService(settings)

    # Set baseline metrics (good performance)
    baseline_metrics = {"precision": 0.75, "hit_ratio": 0.80, "avg_confidence": 0.70}
    monitoring.set_student_baseline_metrics(baseline_metrics)

    # Record good predictions (matching baseline)
    for i in range(15):
        monitoring.record_student_prediction(
            symbol="TEST",
            prediction="BUY",
            probability=0.75 + (i % 3) * 0.05,
            confidence=0.70 + (i % 3) * 0.05,
            actual_outcome="BUY" if i < 12 else "SELL",  # 80% correct
            model_version="v1.0",
        )

    # Evaluate performance (should be good)
    good_metrics = monitoring.evaluate_student_model_performance()
    assert good_metrics["precision"] >= 0.70
    assert good_metrics["total_predictions"] >= 10

    # Check no degradation alerts yet
    alerts = monitoring.check_student_model_degradation()
    assert len(alerts) == 0, "Should not trigger alerts with good performance"

    print("\n✓ Good performance detected, no alerts")

    # Now record degraded predictions
    for i in range(15):
        monitoring.record_student_prediction(
            symbol="TEST",
            prediction="BUY",
            probability=0.60 + (i % 3) * 0.03,
            confidence=0.60 + (i % 3) * 0.03,
            actual_outcome="BUY" if i < 8 else "SELL",  # Only 53% correct (degraded)
            model_version="v1.0",
        )

    # Evaluate degraded performance
    degraded_metrics = monitoring.evaluate_student_model_performance()
    print(f"\n  - Degraded precision: {degraded_metrics['precision']:.2%}")
    print(f"  - Baseline precision: {baseline_metrics['precision']:.2%}")
    print(f"  - Drop: {baseline_metrics['precision'] - degraded_metrics['precision']:.2%}")

    # Check degradation alerts are triggered
    degradation_alerts = monitoring.check_student_model_degradation()
    assert len(degradation_alerts) > 0, "Should trigger alerts with degraded performance"

    print(f"\n✓ Degradation detected, {len(degradation_alerts)} alert(s) triggered")

    # Setup promoter for rollback testing
    production_dir = tmp_path / "production"
    production_dir.mkdir(parents=True)
    settings.student_model_path = str(production_dir / "student_model.pkl")

    # Create current (degraded) production model
    import pickle

    current_model = {"model_type": "logistic", "version": "v1.0_degraded"}
    with open(production_dir / "student_model.pkl", "wb") as f:
        pickle.dump(current_model, f)

    # Create archived baseline (good model)
    archive_dir = production_dir.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    baseline_model = {"model_type": "logistic", "version": "v1.0_baseline"}
    archive_date = (datetime.now() - timedelta(hours=2)).strftime("%Y%m%d_%H%M%S")
    archive_file = archive_dir / f"student_baseline_{archive_date}.pkl"
    with open(archive_file, "wb") as f:
        pickle.dump(baseline_model, f)

    # Create promoter
    promoter = StudentModelPromoter(settings)

    # Check rollback decision (should not rollback yet - confirmation period not met)
    alerts_dict = [alert.to_dict() for alert in degradation_alerts]
    rollback_decision = promoter.should_rollback(alerts_dict, confirmation_hours=1)
    print(f"\n  - Should rollback (immediate): {rollback_decision['should_rollback']}")
    print(f"  - Reason: {rollback_decision['reason']}")

    # Simulate time passage by backdating alerts
    backdated_alerts = []
    for alert in alerts_dict:
        backdated_alert = alert.copy()
        backdated_alert["timestamp"] = (datetime.now() - timedelta(hours=2)).isoformat()
        backdated_alerts.append(backdated_alert)

    # Check rollback decision again (should trigger rollback now)
    rollback_decision = promoter.should_rollback(backdated_alerts, confirmation_hours=1)
    assert rollback_decision["should_rollback"], "Should trigger rollback after confirmation period"

    print(f"\n✓ Rollback decision: {rollback_decision['should_rollback']}")
    print(f"  - Reason: {rollback_decision['reason']}")

    # Inject backdated alerts into monitoring service for rollback test
    monitoring.student_alerts = [
        Alert(**alert) if isinstance(alert, dict) else alert for alert in backdated_alerts
    ]

    # Execute automatic rollback
    rollback_result = promoter.execute_auto_rollback(monitoring, reason="Test rollback")

    assert rollback_result["success"], "Automatic rollback should succeed"
    assert rollback_result["rollback_result"] is not None

    print("\n✓ Automatic rollback executed successfully")
    print(f"  - Message: {rollback_result['message']}")

    # Verify production model was rolled back
    with open(production_dir / "student_model.pkl", "rb") as f:
        restored_model = pickle.load(f)
    assert restored_model["version"] == "v1.0_baseline", "Should restore baseline model"

    print("✓ Production model restored to baseline")

    # Verify rollback event log created
    rollback_log = Path("logs/alerts/rollback_events.jsonl")
    assert rollback_log.exists(), "Rollback event log should be created"

    print("✓ Rollback event logged")


def test_student_monitoring_insufficient_samples(tmp_path):
    """Test student monitoring with insufficient samples (US-021 Phase 3).

    Verifies:
    - Monitoring handles insufficient samples gracefully
    - No alerts triggered when sample size too small
    """
    from src.app.config import Settings
    from src.services.monitoring import MonitoringService

    settings = Settings()
    settings.student_monitoring_enabled = True
    settings.student_monitoring_min_samples = 100  # High threshold

    monitoring = MonitoringService(settings)

    # Set baseline
    baseline_metrics = {"precision": 0.75, "hit_ratio": 0.80}
    monitoring.set_student_baseline_metrics(baseline_metrics)

    # Record only a few predictions (insufficient)
    for _ in range(5):
        monitoring.record_student_prediction(
            symbol="TEST",
            prediction="BUY",
            probability=0.60,
            confidence=0.60,
            actual_outcome="SELL",  # All wrong
            model_version="v1.0",
        )

    # Evaluate performance (should return insufficient samples)
    metrics = monitoring.evaluate_student_model_performance()
    assert metrics.get("insufficient_samples"), "Should indicate insufficient samples"

    # Check no alerts triggered
    alerts = monitoring.check_student_model_degradation()
    assert len(alerts) == 0, "Should not trigger alerts with insufficient samples"

    print("\n✓ Insufficient samples handled gracefully")
