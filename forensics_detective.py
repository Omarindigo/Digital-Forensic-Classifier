import os
import cv2

from rules import (
    rule1_metadata,
    rule2_dhash_whole,
    rule3_dhash_center_crop,
    rule4_tiny_compare,
    _load_gray,
    _dhash,
    _center_crop,
)

class SimpleDetective:

    def __init__(self):
        self.targets = {}

    def register_targets(self, folder):
        print(f"Loading targets from: {folder}")

        for filename in os.listdir(folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(folder, filename)
                file_size = os.path.getsize(filepath)

                gray = _load_gray(filepath)

                dh_whole = _dhash(gray)
                dh_crop75 = _dhash(_center_crop(gray, 0.75))
                dh_crop50 = _dhash(_center_crop(gray, 0.50))
                dh_crop25 = _dhash(_center_crop(gray, 0.25))

                # tiny thumbnails for target crops
                tiny_keep = {}
                for k in [0.75, 0.5, 0.25]:
                    tgray = _center_crop(gray, keep=k)
                    tiny_keep[k] = cv2.resize(tgray, (32, 32), interpolation=cv2.INTER_AREA)

                self.targets[filename] = {
                    'path': filepath,
                    'size': file_size,
                    'dhash_whole': dh_whole,
                    'dhash_crop75': dh_crop75,
                    'dhash_crop50': dh_crop50,
                    'dhash_crop25': dh_crop25,
                    'tiny_keep': tiny_keep
                }

                print(f"  Registered: {filename} ({file_size} bytes)")

        print(f"Total targets: {len(self.targets)}")

    def find_best_match(self, input_image_path):
        print(f"\nProcessing: {os.path.basename(input_image_path)}")
        results = []

        for target_name, target_info in self.targets.items():

            s1, _, e1 = rule1_metadata(target_info, input_image_path)
            s2, _, e2 = rule2_dhash_whole(target_info, input_image_path)
            s3, _, e3 = rule3_dhash_center_crop(target_info, input_image_path)
            s4, _, e4 = rule4_tiny_compare(target_info, input_image_path)

            # keep raw crop score for a simple "strong crop match" check
            s3_raw = s3  # rule3 is 0..5

            # keep total around /30 (simple re-weight)
            s1 = int(s1 * 0.5)         # 0..5-ish
            s2 = int(s2 * (10/15))     # 0..10-ish
            # s3 stays 0..5
            # s4 stays 0..5

            total = s1 + s2 + s3 + s4

            results.append({
                "target": target_name,
                "score": total,
                "s3_raw": s3_raw,
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "e4": e4
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        best = results[0]

        print(f"  Rule 1: {best['e1']}")
        print(f"  Rule 2: {best['e2']}")
        print(f"  Rule 3: {best['e3']}")
        print(f"  Rule 4: {best['e4']}")
        print(f"  Total score: {best['score']}/30")

        # Expert system behavior:
        # If crop rule is perfect (5/5), trust it even if other rules are weak.
        if best["s3_raw"] >= 5:
            print(f"Final: MATCH (strong crop) -> {best['target']}")
            return {"best_match": best['target'], "confidence": best['score']}

        # Normal threshold 
        if best["score"] >= 15:
            print(f"Final: MATCH -> {best['target']}")
            return {"best_match": best['target'], "confidence": best['score']}
        else:
            print("Final: REJECTED")
            return {"best_match": None, "confidence": best['score']}


if __name__ == "__main__":
    print("="*50)
    print("SimpleDetective - Prototype v0.6")
    print("="*50)

    detective = SimpleDetective()
    detective.register_targets("originals")

    print("\n" + "="*50)
    print("TESTING")
    print("="*50)

    # test_images = [
    #     "modified_images/modified_00_bright_enhanced.jpg",
    #     "modified_images/modified_03_compressed.jpg",
    #     "modified_images/modified_02_format_png.png",
    #     "modified_images/modified_01_crop_75pct.jpg",
    #     "modified_images/modified_01_crop_50pct.jpg",
    #     "modified_images/modified_01_crop_25pct.jpg",
    # ]

    # test_images = []

    # for fn in os.listdir("modified_images"):
    #     if fn.lower().endswith((".jpg", ".jpeg", ".png")):
    #         test_images.append(os.path.join("modified_images", fn))

    TEST_FOLDER = "modified_images"
    # choose which folder to test
    # TEST_FOLDER = "hard"
    # TEST_FOLDER = "random"

    
    print("\nChoose test folder:")
    print("1 - modified_images")
    print("2 - hard")
    print("3 - random")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        TEST_FOLDER = "modified_images"
    elif choice == "2":
        TEST_FOLDER = "hard"
    elif choice == "3":
        TEST_FOLDER = "random"
    else:
        print("Invalid choice, defaulting to modified_images")
        TEST_FOLDER = "modified_images"

    print(f"\nTesting folder: {TEST_FOLDER}")

    test_images = []

    for fn in os.listdir(TEST_FOLDER):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            test_images.append(os.path.join(TEST_FOLDER, fn))

    for img in test_images:
        if os.path.exists(img):
            detective.find_best_match(img)

    print("\n" + "="*50)
    print("PROTOTYPE COMPLETE!")
    print("="*50)