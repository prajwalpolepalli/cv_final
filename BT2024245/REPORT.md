Traffic Rule Violation Detection Report

1. Objective
- Detect two-wheelers and riders in street images
- Flag violations: >2 riders, missing helmet, or both
- Extract license plate text for violating vehicles

2. System Overview
- Detector: YOLOv8n (persons, motorcycles)
- Helmet detector: custom YOLOv8n (no-helmet / good-helmet / bad-helmet)
- Plate detector: custom YOLOv8n (number plate)
- OCR: EasyOCR (offline English models)
- Rider association: overlap + position heuristics

3. Model Artifacts and Size
- yolov8n.pt (COCO pretrained)
- helmet_model.pt (custom YOLOv8n)
- plate_model.pt (custom YOLOv8n)
- EasyOCR English models stored in models/easyocr
- Total under 250 MB cap

4. Datasets (used or recommended)
- Indian Helmet Detection Dataset (Roboflow export, YOLOv8)
- Plates derived from the same dataset (numberPlate class)
- Person/motorcycle: COCO pretrained weights

5. Experiments
| ID | Task | Images | Precision | Recall | mAP50 | mAP50-95 | Speed | Notes |
|----|------|--------|-----------|--------|-------|----------|-------|-------|
| E1 | Helmet detector val | 142 | 0.491 | 0.489 | 0.497 | 0.22 | ~170.6ms inf/img | YOLO val (helmet_model_10e-2) |
| E2 | Plate detector val | 117 | 0.715 | 0.612 | 0.647 | 0.221 | ~118.1ms inf/img | YOLO val (plate_model_5ep) |

6. Failure Cases
| Case | Description | Root Cause | Fix Proposal |
|------|-------------|------------|--------------|
| F1 | Bad-helmet class low recall | Few training samples, class imbalance | Add more labeled bad-helmet samples or re-balance classes |
| F2 | Small or blurred plates | Low resolution plates in long shots | Train plate model with higher-res crops or multi-scale aug |

7. Robustness Notes
- Handles multiple motorcycles in a single frame
- Works offline after weights are stored locally (YOLO + EasyOCR)
- Plate detector narrows OCR region for better text extraction

8. Future Improvements
- Add helmet-specific detector to replace face-visibility proxy
- Add dedicated license plate detector for small plates
- Apply tracking/temporal fusion for video inputs

9. Repro Steps
1) Install dependencies: pip install -r requirements.txt
2) Load detector: model = TrafficViolationDetector(model_dir="./models")
3) Run: model.predict("/path/to/image.jpg")

Appendix: Outputs
- Helmet val outputs: runs/detect/val-3
- Plate val outputs: runs/detect/val-4
- Helmet predictions: runs/detect/runs/helmet/predictions_10e
