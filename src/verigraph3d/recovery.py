from __future__ import annotations

from dataclasses import replace

from .models import ActionRecord, ActionType, FailureAttribution, SceneState, SemanticAction, VerificationReport


class FailureAnalyzer:
    def attribute(self, record: ActionRecord, report: VerificationReport) -> FailureAttribution | None:
        if record.status == "failed":
            kind = "execution_error"
            if record.error and "Preflight" in record.error:
                kind = "precondition_or_parameter_error"
            if record.error_details.get("source") == "agentblender_world_state":
                kind = "agentblender_transaction_rejected"
            evidence = {"error": record.error, **record.error_details}
            failed_constraint = str(record.error_details.get("failed_constraint", "action_execution"))
            return FailureAttribution(record.action.id, kind, failed_constraint, [record.action.target], evidence, "rollback_or_replan")
        failed_results = [r for r in report.deterministic if not r.passed]
        failed = next((r for r in failed_results if r.code == "geometry_intersection"), None)
        failed = failed or (failed_results[0] if failed_results else None)
        if failed is None and report.visual_score is not None and report.visual_score < 0.8:
            return FailureAttribution(record.action.id, "visual_mismatch", "visual_similarity", [record.action.target], {"score": report.visual_score}, "adjust_appearance_or_camera")
        if failed is None:
            return None
        objects = failed.evidence.get("objects", [record.action.target])
        recovery = "increase_z" if failed.code == "geometry_intersection" else "local_replan"
        return FailureAttribution(record.action.id, "constraint_solver_error", failed.code, objects, failed.evidence, recovery)


class LocalRepairer:
    def propose(self, action: SemanticAction, failure: FailureAttribution, current: SceneState | None = None, step: float = 0.02) -> SemanticAction | None:
        if failure.recommended_recovery == "increase_z" and "location" in action.parameters:
            actual = current.objects.get(action.target).location if current and action.target in current.objects else action.parameters["location"]
            location = list(actual)
            depth = failure.evidence.get("penetration_depth", step)
            location[2] += max(step, float(depth))
            return replace(action, id=f"{action.id}_repair", parameters={**action.parameters, "location": tuple(location)})
        if failure.recommended_recovery == "adjust_appearance_or_camera" and action.type == ActionType.SET_CAMERA:
            return replace(action, id=f"{action.id}_repair")
        return None
