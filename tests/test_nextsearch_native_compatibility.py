"""Small stdlib regression tests; no model, network, or GPU required."""
import unittest

from scripts.audit_nextsearch_native_compatibility import (
    action_compatibility, summarize,
)


def call(name, args):
    return {"function": {"name": name, "arguments": args}}


def row(state, target):
    return {
        "task_id": "independent-task:" + state,
        "source_parent_sha256": "same-original-parent",
        "source_split": "train",
        "capability": state,
        "original_teacher_action_unchanged": True,
        "task_success_independently_replayed": False,
        "training_action_indices": [3],
        "messages": [
            {"role": "user", "content": "Research"},
            {"role": "assistant", "tool_calls": [call("search", {"objective": "Research", "search_queries": ["A", "B"]})]},
            {"role": "tool", "content": "an actual original observation"},
            target,
        ],
    }


class NativeSourceGateTests(unittest.TestCase):
    def test_batch_source_action_cannot_be_silent_native_single_call(self):
        self.assertEqual(
            action_compatibility(call("search", {"objective": "a", "search_queries": ["a", "b"]})),
            "batched_search",
        )
        self.assertEqual(
            action_compatibility(call("fetch", {"objective": "a", "urls": ["https://a.example", "https://b.example"]})),
            "batched_fetch",
        )

    def test_single_source_call_is_only_syntax_candidate(self):
        self.assertEqual(
            action_compatibility(call("search", {"objective": "a", "search_queries": ["a"]})),
            "one_query_syntax_only",
        )
        self.assertEqual(
            action_compatibility(call("fetch", {"urls": ["https://a.example"], "offset": 0})),
            "one_url_syntax_only",
        )
        self.assertEqual(
            action_compatibility(call("fetch", {"urls": ["https://a.example"], "offset": 10})),
            "fetch_offset_not_proven_equivalent",
        )

    def test_original_source_evidence_does_not_become_native_qlora_supervision(self):
        rows = [
            row("evidence_missing_continue", {"role": "assistant", "tool_calls": [call("search", {"objective": "x", "search_queries": ["x"]})]}),
            row("evidence_present_finalize", {"role": "assistant", "content": "answer"}),
        ]
        report = summarize(rows)
        self.assertEqual(report["states"]["evidence_present_finalize"], 1)
        self.assertEqual(report["target_actions"]["one_query_syntax_only"], 1)
        self.assertEqual(report["history"]["fully_native_schema_history"], 0)
        self.assertEqual(report["native_qlora_training_examples_certified"], 0)
        self.assertIsNone(report["new_gaia_score"])

    def test_rejects_protected_validation_source(self):
        bad = row("evidence_present_finalize", {"role": "assistant", "content": "answer"})
        bad["source_split"] = "validation"
        with self.assertRaisesRegex(ValueError, "protected source split"):
            summarize([bad])


if __name__ == "__main__":
    unittest.main()
