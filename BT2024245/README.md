Traffic Rule Violation Detection (AID 728)

Project contents
- solution.py: TrafficViolationDetector implementation
- models/: pretrained weights (yolov8n + RapidOCR ONNX)
- requirements.txt: Python dependencies

Setup
1) Create and activate a Python 3.10+ environment
2) Install dependencies:
   pip install -r requirements.txt

Usage
from solution import TrafficViolationDetector

model = TrafficViolationDetector(model_dir="./models")
output = model.predict("/path/to/image.jpg")
print(output)

Output format
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 2,
      "license_plate": "TN01AB1234"
    }
  ]
}

Pipeline summary
1) YOLOv8n detects persons and motorcycles.
2) Riders are associated to motorcycles by overlap and position.
3) Helmet violations are estimated from face visibility in the head region.
4) License plates are localized with contour heuristics and recognized using RapidOCR.

Datasets (recommended for analysis/extension)
- Helmet: Safety Helmet Wearing Dataset (SHWD)
- License plates: CCPD, UFPR-ALPR
- Riders/vehicles: MS-COCO, BDD100K

Notes
- All models are loaded in the constructor as required.
- Inference is stateless and offline-ready with local weights.
- You can swap in custom detectors by replacing model files in models/.
