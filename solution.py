"""
solution.py — Traffic Rule Violation Detection
AID 728 Course Project

Combined:
  - Helmet / rider / violation detection  →  from solution1.py
  - License plate detection + OCR         →  from solution.py

Models (all inside model_dir, total < 250 MB):
  yolov8n.pt
  helmet_model.pt
  plate_model.pt
  rapidocr/ch_PP-OCRv3_det_infer.onnx
  rapidocr/ch_PP-OCRv3_rec_infer.onnx
  rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx
"""

from pathlib import Path

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR   # must come before YOLO
except ImportError as exc:
    raise ImportError(
        "RapidOCR is required. Install it with 'pip install rapidocr-onnxruntime'."
    ) from exc

from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers  (solution.py style — used for plate association)
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a, b):
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────

class TrafficViolationDetector:

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)

        # ── 1. OCR first — prevents MKL/ONNXRuntime vs PyTorch conflict ──────
        ocr_dir  = self.model_dir / "rapidocr"
        det_path = ocr_dir / "ch_PP-OCRv3_det_infer.onnx"
        rec_path = ocr_dir / "ch_PP-OCRv3_rec_infer.onnx"
        cls_path = ocr_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx"

        if not det_path.exists() or not rec_path.exists() or not cls_path.exists():
            raise FileNotFoundError(
                "RapidOCR ONNX files not found. Expected files in ./models/rapidocr."
            )

        self.ocr = RapidOCR(
            det_model_path=str(det_path),
            rec_model_path=str(rec_path),
            cls_model_path=str(cls_path),
            use_cuda=False,
        )
        # Warm-up: fully initialises ONNXRuntime before PyTorch loads
        _ = self.ocr(np.zeros((64, 64, 3), dtype=np.uint8))

        # ── 2. YOLO models — loaded after OCR warm-up ────────────────────────
        for name in ("yolov8n.pt", "helmet_model.pt", "plate_model.pt"):
            if not (self.model_dir / name).exists():
                raise FileNotFoundError(f"Missing model file: {self.model_dir / name}")

        self.detector        = YOLO(str(self.model_dir / "yolov8n.pt"))
        self.helmet_detector = YOLO(str(self.model_dir / "helmet_model.pt"))
        self.plate_detector  = YOLO(str(self.model_dir / "plate_model.pt"))

        # Helmet model class IDs
        self.helmet_no_id   = 0   # no helmet
        self.helmet_good_id = 1   # helmet worn correctly
        self.helmet_bad_id  = 2   # helmet worn incorrectly / partial

    # ─────────────────────────────────────────────────────────────────────────
    #  predict()
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            return {"violations": []}

        img_h, img_w = image.shape[:2]

        # ── Stage 1 & 2: persons, motorcycles, helmets  (solution1.py) ───────
        persons, motorcycles = self._detect_persons_and_motorcycles(image)
        helmet_dets          = self._detect_helmets(image)

        # ── Stage 3: plate detections on full image  (solution.py) ────────────
        # Detect all plates at once, run OCR immediately, store with coords
        all_plates = self._detect_plates_with_ocr(image, img_h, img_w)

        # ── Stage 4: assign riders to bikes  (solution1.py) ───────────────────
        assignments = self._assign_riders_to_motorcycles(persons, motorcycles)

        # ── Stage 5: build violations ─────────────────────────────────────────
        violations = []
        for moto in motorcycles:
            rider_ids = assignments.get(moto["id"], [])
            if not rider_ids:
                continue

            riders            = [persons[idx] for idx in rider_ids]
            helmet_violations = self._count_helmet_violations(riders, helmet_dets)
            num_riders        = len(riders)

            if num_riders > 2 or helmet_violations > 0:
                # ── Plate: solution.py's IoU-based association ────────────────
                license_plate = self._best_plate_for_moto(
                    moto["bbox"], all_plates, img_h
                )

                violations.append({
                    "num_riders":        int(num_riders),
                    "helmet_violations": int(helmet_violations),
                    "license_plate":     license_plate,
                })

        return {"violations": violations}

    # ─────────────────────────────────────────────────────────────────────────
    #  Detection — persons & motorcycles  (solution1.py)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_persons_and_motorcycles(self, image):
        results = self.detector(image, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        result  = results[0]
        persons, motorcycles = [], []
        for idx, box in enumerate(result.boxes):
            cls_id          = int(box.cls[0])
            label           = result.names.get(cls_id, "")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            entry = {
                "id":   idx,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(box.conf[0]),
            }
            if label == "person":
                persons.append(entry)
            elif label == "motorcycle":
                motorcycles.append(entry)
        return persons, motorcycles

    def _detect_helmets(self, image):
        results = self.helmet_detector(image, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        result  = results[0]
        detections = []
        for box in result.boxes:
            cls_id          = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((cls_id, int(x1), int(y1), int(x2), int(y2)))
        return detections

    # ─────────────────────────────────────────────────────────────────────────
    #  Detection — plates + OCR  (solution.py approach)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_plates_with_ocr(self, image, img_h, img_w):
        """
        Detect all plates in the full image at once, immediately OCR each crop.
        Returns list of ([x1,y1,x2,y2], text_string).
        This is solution.py's approach — better recall than cropping per-bike.
        """
        results = self.plate_detector(image, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        result  = results[0]
        plates  = []

        for box in result.boxes:
            if float(box.conf[0]) < 0.25:
                continue
            px1, py1, px2, py2 = [int(v) for v in box.xyxy[0]]
            px1 = max(0, px1);  py1 = max(0, py1)
            px2 = min(img_w, px2);  py2 = min(img_h, py2)

            crop = image[py1:py2, px1:px2]
            text = self._ocr_full(crop)
            plates.append(([px1, py1, px2, py2], text))

        return plates

    def _best_plate_for_moto(self, moto_bbox, all_plates, img_h):
        """
        Find best matching plate for a motorcycle using IoU on a
        downward-expanded box (solution.py's approach).
        Falls back to solution1.py's contour method if no plate matches.
        """
        mx1, my1, mx2, my2 = moto_bbox

        # Expand downward to catch plates mounted below the bike frame
        expanded = [mx1, my1, mx2, min(img_h, my2 + 80)]

        best_text  = "N/A"
        best_score = 0.0
        for pb, pt in all_plates:
            score = _iou(pb, expanded)
            if score > best_score:
                best_score = score
                best_text  = pt if pt else "unreadable"

        return best_text

    def _ocr_full(self, crop: np.ndarray) -> str:
        """
        solution.py's OCR approach — direct RapidOCR on raw crop,
        returns cleaned uppercase text.
        """
        if crop.size == 0:
            return ""
        try:
            result, _ = self.ocr(crop)
            if not result:
                return ""
            text = " ".join(r[1] for r in result if r[1]).strip()
            return text.replace(" ", "").upper()
        except Exception:
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Rider assignment  (solution1.py — bottom-centre point method)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _box_inside(inner, outer, thresh=0.30):
        """Return True if >= thresh fraction of inner's area lies inside outer."""
        xi1 = max(inner[0], outer[0]); yi1 = max(inner[1], outer[1])
        xi2 = min(inner[2], outer[2]); yi2 = min(inner[3], outer[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area  = (inner[2] - inner[0]) * (inner[3] - inner[1]) + 1e-6
        return (inter / area) >= thresh

    def _assign_riders_to_motorcycles(self, persons, motorcycles):
        assignments = {moto["id"]: [] for moto in motorcycles}
        if not motorcycles:
            return assignments

        for p_idx, person in enumerate(persons):
            px1, py1, px2, py2 = person["bbox"]

            # Method 1 (visualize1.py): bottom-centre point of person
            px = int((px1 + px2) / 2)
            py = int(py2)

            best_moto, best_score = None, 0.0
            for moto in motorcycles:
                mx1, my1, mx2, my2 = moto["bbox"]
                mw, mh = mx2 - mx1, my2 - my1

                ex1 = mx1 - int(mw * 0.15);  ex2 = mx2 + int(mw * 0.15)
                ey1 = my1 - int(mh * 0.20);  ey2 = my2 + int(mh * 0.20)

                # Method 1: bottom-centre inside expanded box (visualize1.py)
                bottom_centre_match = (ex1 <= px <= ex2 and ey1 <= py <= ey2)

                # Method 2: 30% area overlap (visualize.py)
                expanded_box = (ex1, ey1, ex2, ey2)
                area_overlap_match = self._box_inside(
                    (px1, py1, px2, py2), expanded_box, thresh=0.30
                )

                if bottom_centre_match or area_overlap_match:
                    overlap = self._bbox_iou(person["bbox"], moto["bbox"])
                    if overlap > best_score:
                        best_score, best_moto = overlap, moto

            if best_moto is not None:
                assignments[best_moto["id"]].append(p_idx)

        return assignments

    # ─────────────────────────────────────────────────────────────────────────
    #  Helmet logic  (solution1.py — head-region check)
    # ─────────────────────────────────────────────────────────────────────────

    def _count_helmet_violations(self, riders, helmet_dets):
        violations = 0
        for rider in riders:
            if self._helmet_status_for_rider(rider["bbox"], helmet_dets) != "good":
                violations += 1
        return violations

    def _helmet_status_for_rider(self, rider_bbox, helmet_dets):
        """Check only the top 40% of the rider box (head region)."""
        x1, y1, x2, y2 = rider_bbox
        head_box = (x1, y1, x2, y1 + int((y2 - y1) * 0.4))

        has_good = has_violation = False
        for cls_id, hx1, hy1, hx2, hy2 in helmet_dets:
            if self._bbox_iou(head_box, (hx1, hy1, hx2, hy2)) < 0.1:
                continue
            if cls_id == self.helmet_good_id:
                has_good = True
            elif cls_id in (self.helmet_no_id, self.helmet_bad_id):
                has_violation = True

        if has_violation:
            return "violation"
        if has_good:
            return "good"
        return "unknown"

    # ─────────────────────────────────────────────────────────────────────────
    #  Geometry utility
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom  = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Quick local test
#  python solution.py --image path/to/image.jpg --model_dir ./models
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--model_dir", default="./models")
    args = parser.parse_args()

    detector = TrafficViolationDetector(model_dir=args.model_dir)
    output   = detector.predict(args.image)
    print(json.dumps(output, indent=2))