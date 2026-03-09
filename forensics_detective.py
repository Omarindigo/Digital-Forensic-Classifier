import os
from typing import Dict, Any, List, Optional


class SimpleDetective:
    def __init__(self, rules_module):
        self.targets: Dict[str, Dict[str, Any]] = {}
        self.rules = rules_module

    def register_targets(self, folder: str) -> None:
        self.targets = {}
        print(f"Loading originals from: {folder}")

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            filepath = os.path.join(folder, filename)
            signature = self.rules.build_target_signature(filepath)

            self.targets[filename] = {
                "filename": filename,
                "path": filepath,
                **signature,
            }

            print(f"Registered target: {filename}")

        print(f"Total targets loaded: {len(self.targets)}")

    def find_best_match(self, input_image_path: str) -> Dict[str, Any]:
        input_signature = self.rules.build_input_signature(input_image_path)
        input_name = os.path.basename(input_image_path)

        candidate_results: List[Dict[str, Any]] = []

        for target_name, target_info in self.targets.items():
            r1 = self.rules.rule1_metadata(target_info, input_signature)
            r2 = self.rules.rule2_histogram(target_info, input_signature)
            r3 = self.rules.rule3_template(target_info, input_signature)

            total_score = r1["score"] + r2["score"] + r3["score"]
            rules_used = [r1, r2, r3]

            if hasattr(self.rules, "rule4_extra"):
                r4 = self.rules.rule4_extra(target_info, input_signature)
                total_score += r4["score"]
                rules_used.append(r4)

            candidate_results.append({
                "target": target_name,
                "score": total_score,
                "rules": rules_used
            })

        candidate_results.sort(key=lambda x: x["score"], reverse=True)
        best = candidate_results[0]

        threshold = getattr(self.rules, "MATCH_THRESHOLD", 50)
        matched = best["score"] >= threshold

        lines = [f"Processing: {input_name}"]
        for rule_result in best["rules"]:
            lines.append(rule_result["line"])

        if matched:
            lines.append(f"Final Score: {best['score']}/100 -> MATCH to {best['target']}")
        else:
            lines.append(f"Final Score: {best['score']}/100 -> REJECTED")

        return {
            "input": input_name,
            "best_match": best["target"] if matched else None,
            "matched": matched,
            "confidence": best["score"],
            "rules": best["rules"],
            "output_text": "\n".join(lines),
        }

    def process_folder(self, folder: str, ground_truth: bool = True) -> Dict[str, Any]:
        image_paths = []

        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder, filename))

        outputs: List[str] = []
        results: List[Dict[str, Any]] = []

        print(f"\nProcessing folder: {folder}")
        print(f"Total files found: {len(image_paths)}")

        for i, path in enumerate(image_paths, start=1):
            print(f"[{i}/{len(image_paths)}] Processing file: {os.path.basename(path)}")
            result = self.find_best_match(path)
            results.append(result)
            outputs.append(result["output_text"])

        report_text = "\n\n".join(outputs)

        summary = None
        if ground_truth:
            summary = self.evaluate_results(results, folder)

        return {
            "results": results,
            "report_text": report_text,
            "summary": summary
        }

    def evaluate_results(self, results: List[Dict[str, Any]], folder_name: str) -> Dict[str, Any]:
        folder_lower = folder_name.lower()

        if "random" in folder_lower:
            total = len(results)
            rejected = sum(1 for r in results if not r["matched"])
            false_positives = total - rejected
            false_positive_rate = (false_positives / total * 100.0) if total else 0.0

            return {
                "type": "random",
                "total": total,
                "rejected": rejected,
                "false_positives": false_positives,
                "false_positive_rate": false_positive_rate,
            }

        total = len(results)
        correct = 0

        for r in results:
            true_prefix = self.extract_true_original_prefix(r["input"])
            predicted_prefix = self.extract_true_original_prefix(r["best_match"]) if r["best_match"] else None

            if true_prefix is not None and predicted_prefix is not None and true_prefix == predicted_prefix:
                correct += 1

        accuracy = (correct / total * 100.0) if total else 0.0

        return {
            "type": "derived",
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
        }

    @staticmethod
    def extract_true_original_prefix(filename: Optional[str]) -> Optional[str]:
        if not filename:
            return None

        name = os.path.basename(filename).lower()

        if "original_" in name:
            idx = name.find("original_")
            suffix = name[idx + len("original_"):]
            digits = ""
            for ch in suffix:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                return f"original_{digits}.jpg"

        if "modified_" in name:
            idx = name.find("modified_")
            suffix = name[idx + len("modified_"):]
            digits = ""
            for ch in suffix:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                return f"original_{digits}.jpg"

        return None