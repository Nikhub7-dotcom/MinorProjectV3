"""
Generates configs/experiments/*.yaml for the full grid: every model in
MODEL_NAMES x every rate in RATES, for the backdoor attack (the attack
type most worth comparing across architectures, since ASR + clean
accuracy together show whether a bigger/different model is harder or
easier to backdoor).

Run once after cloning:
    python src/generate_configs.py

Re-run any time to regenerate — it always overwrites, so hand-edits to
generated files won't survive a re-run; edit RATES/MODEL_NAMES/ATTACK
below instead, or copy a generated file to a new name if you want a
one-off variant.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry_models import MODEL_NAMES

RATES = [0.001, 0.005, 0.01]  # 0.1%, 0.5%, 1%
ATTACK = "backdoor"
TARGET_CLASS = 0
EPOCHS_BY_MODEL = {
    # Heavier architectures get fewer default epochs to keep a full
    # sweep finishing in reasonable time — bump these yourself for a
    # final run once you've smoke-tested each one.
    "resnet10": 60, "resnet18": 60, "resnet50": 40, "vgg19": 40, "densenet161": 30,
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "experiments")


def rate_tag(rate):
    return str(rate).replace(".", "p")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for model in MODEL_NAMES:
        for rate in RATES:
            name = f"cifar10_{model}_{ATTACK}_{rate_tag(rate)}.yaml"
            path = os.path.join(OUT_DIR, name)
            content = f"""dataset: cifar10
model: {model}
attack: {ATTACK}
rate: {rate}
target_class: {TARGET_CLASS}
trigger_size: 3
trigger_position: bottom_right
trigger_color: [255, 255, 255]
epochs: {EPOCHS_BY_MODEL[model]}
batch_size: 128
run_detectors: true
max_minutes: 170
"""
            with open(path, "w") as f:
                f.write(content)
            written.append(name)

    print(f"[generate_configs] wrote {len(written)} configs to {OUT_DIR}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
