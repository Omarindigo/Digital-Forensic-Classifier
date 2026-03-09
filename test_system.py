import os

from forensics_detective import SimpleDetective
import rules
import rules_v2


ORIGINALS_FOLDER = "originals"
MODIFIED_FOLDER = "modified_images"
HARD_FOLDER = "hard"
RANDOM_FOLDER = "random"


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_summary_block(text: str, title: str, summary: dict) -> str:
    if summary is None:
        return text

    lines = [text, "", title]

    if summary["type"] == "derived":
        lines.append(f"Accuracy: {summary['correct']}/{summary['total']} ({summary['accuracy']:.2f}%)")
    elif summary["type"] == "random":
        lines.append(
            f"False Positive Rate: {summary['false_positives']}/{summary['total']} "
            f"({summary['false_positive_rate']:.2f}%)"
        )

    return "\n".join(lines)


def run_v1():
    print("\nStarting V1...")
    detective = SimpleDetective(rules)

    print("Registering originals for V1...")
    detective.register_targets(ORIGINALS_FOLDER)

    print("\nRunning V1 on modified_images...")
    modified_report = detective.process_folder(MODIFIED_FOLDER, ground_truth=True)

    print("\nRunning V1 on random...")
    random_report = detective.process_folder(RANDOM_FOLDER, ground_truth=True)

    combined_text = modified_report["report_text"] + "\n\n" + random_report["report_text"]
    combined_text = append_summary_block(combined_text, "V1 Summary (Modified Images)", modified_report["summary"])
    combined_text = append_summary_block(combined_text, "V1 Summary (Random Images)", random_report["summary"])
    save_text("results_v1.txt", combined_text)
    print("Saved results_v1.txt")

    print("\nRunning V1 on hard...")
    hard_report = detective.process_folder(HARD_FOLDER, ground_truth=True)
    hard_text = append_summary_block(hard_report["report_text"], "V1 Summary (Hard Images)", hard_report["summary"])
    save_text("results_v1_hard.txt", hard_text)
    print("Saved results_v1_hard.txt")

    return {
        "modified": modified_report["summary"],
        "random": random_report["summary"],
        "hard": hard_report["summary"],
    }


def run_v2():
    print("\nStarting V2...")
    detective = SimpleDetective(rules_v2)

    print("Registering originals for V2...")
    detective.register_targets(ORIGINALS_FOLDER)

    print("\nRunning V2 on modified_images...")
    modified_report = detective.process_folder(MODIFIED_FOLDER, ground_truth=True)

    print("\nRunning V2 on hard...")
    hard_report = detective.process_folder(HARD_FOLDER, ground_truth=True)

    print("\nRunning V2 on random...")
    random_report = detective.process_folder(RANDOM_FOLDER, ground_truth=True)

    combined_text = (
        modified_report["report_text"]
        + "\n\n"
        + hard_report["report_text"]
        + "\n\n"
        + random_report["report_text"]
    )

    combined_text = append_summary_block(combined_text, "V2 Summary (Modified Images)", modified_report["summary"])
    combined_text = append_summary_block(combined_text, "V2 Summary (Hard Images)", hard_report["summary"])
    combined_text = append_summary_block(combined_text, "V2 Summary (Random Images)", random_report["summary"])
    save_text("results_v2.txt", combined_text)
    print("Saved results_v2.txt")

    return {
        "modified": modified_report["summary"],
        "hard": hard_report["summary"],
        "random": random_report["summary"],
    }


def print_console_summary(v1_stats, v2_stats):
    print("\nFinished generating:")
    print("results_v1.txt")
    print("results_v1_hard.txt")
    print("results_v2.txt")

    print("\nV1:")
    print(f"Modified accuracy: {v1_stats['modified']['accuracy']:.2f}%")
    print(f"Hard accuracy: {v1_stats['hard']['accuracy']:.2f}%")
    print(f"Random false positive rate: {v1_stats['random']['false_positive_rate']:.2f}%")

    print("\nV2:")
    print(f"Modified accuracy: {v2_stats['modified']['accuracy']:.2f}%")
    print(f"Hard accuracy: {v2_stats['hard']['accuracy']:.2f}%")
    print(f"Random false positive rate: {v2_stats['random']['false_positive_rate']:.2f}%")


if __name__ == "__main__":
    required_folders = [ORIGINALS_FOLDER, MODIFIED_FOLDER, HARD_FOLDER, RANDOM_FOLDER]
    missing = [folder for folder in required_folders if not os.path.isdir(folder)]

    if missing:
        print("Missing required folders:")
        for folder in missing:
            print(f"- {folder}")
        raise SystemExit(1)

    v1_stats = run_v1()
    v2_stats = run_v2()
    print_console_summary(v1_stats, v2_stats)