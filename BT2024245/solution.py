from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "RapidOCR is required. Install it with 'pip install rapidocr-onnxruntime'."
    ) from exc


class TrafficViolationDetector:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        yolo_path = self.model_dir / "yolov8n.pt"
        if not yolo_path.exists():
            raise FileNotFoundError(f"Missing model file: {yolo_path}")

        self.detector = YOLO(str(yolo_path))

        helmet_path = self.model_dir / "helmet_model.pt"
        plate_path = self.model_dir / "plate_model.pt"
        if not helmet_path.exists():
            raise FileNotFoundError(f"Missing model file: {helmet_path}")
        if not plate_path.exists():
            raise FileNotFoundError(f"Missing model file: {plate_path}")

        self.helmet_detector = YOLO(str(helmet_path))
        self.plate_detector = YOLO(str(plate_path))

        # Helmet model class IDs (re-mapped during training)
        self.helmet_no_id = 0
        self.helmet_good_id = 1
        self.helmet_bad_id = 2

        ocr_dir = self.model_dir / "rapidocr"
        det_path = ocr_dir / "ch_PP-OCRv3_det_infer.onnx"
        rec_path = ocr_dir / "ch_PP-OCRv3_rec_infer.onnx"
        cls_path = ocr_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        if not det_path.exists() or not rec_path.exists() or not cls_path.exists():
            raise FileNotFoundError(
                "RapidOCR model files not found. Expected ONNX files in ./models/rapidocr."
            )

        self.ocr = RapidOCR(
            det_model_path=str(det_path),
            rec_model_path=str(rec_path),
            cls_model_path=str(cls_path),
            use_cuda=False,
        )

    def predict(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            return {"violations": []}

        persons, motorcycles = self._detect_persons_and_motorcycles(image)
        helmet_dets = self._detect_helmets(image)
        plate_dets = self._detect_plates(image)
        assignments = self._assign_riders_to_motorcycles(persons, motorcycles)

        violations = []
        for moto in motorcycles:
            rider_ids = assignments.get(moto["id"], [])
            if not rider_ids:
                continue

            riders = [persons[idx] for idx in rider_ids]
            helmet_violations = self._count_helmet_violations(image, riders, helmet_dets)
            num_riders = len(riders)

            if num_riders > 2 or helmet_violations > 0:
                license_plate = self._extract_license_plate(
                    image,
                    moto["bbox"],
                    plate_dets,
                )
                violations.append(
                    {
                        "num_riders": int(num_riders),
                        "helmet_violations": int(helmet_violations),
                        "license_plate": license_plate,
                    }
                )

        return {"violations": violations}

    def _detect_persons_and_motorcycles(self, image):
        results = self.detector(image, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        result = results[0]
        boxes = result.boxes

        persons = []
        motorcycles = []
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, "")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            entry = {
                "id": idx,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(box.conf[0]),
            }

            if label == "person":
                persons.append(entry)
            elif label == "motorcycle":
                motorcycles.append(entry)

        return persons, motorcycles

    def _assign_riders_to_motorcycles(self, persons, motorcycles):
        assignments = {moto["id"]: [] for moto in motorcycles}
        if not motorcycles:
            return assignments

        for p_idx, person in enumerate(persons):
            px1, py1, px2, py2 = person["bbox"]
            px = int((px1 + px2) / 2)
            py = int(py2)

            best_moto = None
            best_score = 0.0
            for moto in motorcycles:
                mx1, my1, mx2, my2 = moto["bbox"]
                mw = mx2 - mx1
                mh = my2 - my1

                expand_x = int(mw * 0.15)
                expand_y = int(mh * 0.2)
                ex1 = mx1 - expand_x
                ex2 = mx2 + expand_x
                ey1 = my1 - expand_y
                ey2 = my2 + expand_y

                if ex1 <= px <= ex2 and ey1 <= py <= ey2:
                    overlap = self._bbox_iou(person["bbox"], moto["bbox"])
                    if overlap > best_score:
                        best_score = overlap
                        best_moto = moto

            if best_moto is not None:
                assignments[best_moto["id"]].append(p_idx)

        return assignments

    def _count_helmet_violations(self, image, riders, helmet_dets):
        violations = 0
        for rider in riders:
            status = self._helmet_status_for_rider(rider["bbox"], helmet_dets)
            if status != "good":
                violations += 1
        return violations

    def _extract_license_plate(self, image, moto_bbox, plate_dets):
        x1, y1, x2, y2 = moto_bbox
        moto_roi = image[y1:y2, x1:x2]
        if moto_roi.size == 0:
            return ""

        candidates = self._plate_candidates(moto_roi, plate_dets, moto_bbox)
        best_text = ""
        best_score = 0.0

        for cx1, cy1, cx2, cy2 in candidates:
            plate = moto_roi[cy1:cy2, cx1:cx2]
            text, score = self._ocr_plate(plate)
            if score > best_score:
                best_text = text
                best_score = score

        if not best_text:
            # Fallback: try OCR on the lower half of the motorcycle region
            h = moto_roi.shape[0]
            fallback_roi = moto_roi[int(h * 0.5):, :]
            text, _ = self._ocr_plate(fallback_roi)
            best_text = text

        return best_text

    def _plate_candidates(self, roi, plate_dets, moto_bbox):
        abs_candidates = []
        mx1, my1, mx2, my2 = moto_bbox
        for det in plate_dets:
            px1, py1, px2, py2 = det
            if px1 >= mx1 and py1 >= my1 and px2 <= mx2 and py2 <= my2:
                abs_candidates.append((px1, py1, px2, py2))

        if abs_candidates:
            return [
                (x1 - mx1, y1 - my1, x2 - mx1, y2 - my1)
                for x1, y1, x2, y2 in abs_candidates
            ]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        h, w = gray.shape[:2]
        min_area = max(100, int(0.01 * w * h))

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0:
                continue

            aspect = cw / float(ch)
            if 2.0 <= aspect <= 6.0:
                candidates.append((x, y, x + cw, y + ch))

        if not candidates:
            return [(0, int(h * 0.55), w, h)]

        return candidates

    def _ocr_plate(self, plate_img):
        if plate_img.size == 0:
            return "", 0.0

        prep = self._prepare_for_ocr(plate_img)
        result, _ = self.ocr(prep)
        if not result:
            return "", 0.0

        best_text = ""
        best_score = 0.0
        for _, text, score in result:
            clean_text = "".join(ch for ch in text.upper() if ch.isalnum())
            if len(clean_text) < 4:
                continue
            if score > best_score:
                best_score = float(score)
                best_text = clean_text

        return best_text, best_score

    @staticmethod
    def _prepare_for_ocr(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def _detect_helmets(self, image):
        results = self.helmet_detector(image, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        result = results[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((cls_id, int(x1), int(y1), int(x2), int(y2)))
        return detections

    def _detect_plates(self, image):
        results = self.plate_detector(image, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        result = results[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((int(x1), int(y1), int(x2), int(y2)))
        return detections

    def _helmet_status_for_rider(self, rider_bbox, helmet_dets):
        x1, y1, x2, y2 = rider_bbox
        h = y2 - y1
        head_box = (x1, y1, x2, y1 + int(h * 0.4))

        has_good = False
        has_violation = False
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

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom
