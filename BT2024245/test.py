import argparse
import json
from pathlib import Path

from solution import TrafficViolationDetector


def main():
    parser = argparse.ArgumentParser(description="Run traffic violation detector")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model-dir", default="./models", help="Model directory")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector = TrafficViolationDetector(model_dir=args.model_dir)
    output = detector.predict(str(image_path))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
