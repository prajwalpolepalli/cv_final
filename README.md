# Traffic Rule Violation Detection (AID 728)

An automated computer vision pipeline designed to detect traffic rule violations involving motorcycles. This project specifically identifies instances of **triple riding** (more than 2 riders on a single motorcycle) and **helmetless riding**, while also detecting and reading the **license plates** of the violating vehicles using Optical Character Recognition (OCR).

## 📁 Project Structure

- `solution.py`: Main implementation containing the `TrafficViolationDetector` class and the complete detection pipeline.
- `models/`: Directory containing all the pre-trained local weights:
  - `yolov8n.pt`: Standard YOLOv8 Nano model for person and motorcycle detection.
  - `helmet_model.pt`: Custom-trained YOLO model for helmet detection.
  - `plate_model.pt`: Custom-trained YOLO model for license plate detection.
  - `rapidocr/`: ONNX models used by RapidOCR for character recognition.
- `data/`: Contains datasets and YOLO YAML configuration files (`helmet_yolo.yaml`, `plate_yolo.yaml`, etc.) used for training the custom models.
- `requirements.txt`: Python dependencies required to run the pipeline.
- `report.pdf`: Detailed project report containing methodology, experiments, and results.

## 🚀 Setup & Installation

1. **Prerequisites:** Ensure you have Python 3.10+ installed.
2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This installs `ultralytics`, `opencv-python`, `torch`, `rapidocr-onnxruntime`, etc.)*

## 💻 Usage

### Python API
You can easily integrate the detector into your own Python scripts:

```python
from solution import TrafficViolationDetector

# Initialize the detector (loads models from the specified directory)
detector = TrafficViolationDetector(model_dir="./models")

# Run prediction on an image
output = detector.predict("/path/to/image.jpg")
print(output)
```

### Command Line Interface (CLI)
You can also run the detector directly from the command line on a single image:

```bash
python solution.py --image path/to/image.jpg --model_dir ./models
```

### Output Format
The pipeline returns a JSON-friendly dictionary detailing the violations found per motorcycle. If a motorcycle has >2 riders OR any rider is missing a helmet, the license plate is read.

```json
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 2,
      "license_plate": "TN01AB1234"
    }
  ]
}
```
*Note: If the license plate is obscured or not found, `license_plate` might be returned as `"N/A"` or `"unreadable"`.*

## 🔍 Pipeline Architecture

The detection pipeline operates in a stateless, offline-ready manner through the following stages:

1. **Object Detection:** `yolov8n` detects `person` and `motorcycle` classes across the image.
2. **Helmet Detection:** The custom `helmet_model.pt` detects heads/helmets and classifies them into: good (worn correctly), bad (worn incorrectly), or none (no helmet).
3. **Plate Detection & OCR:** The custom `plate_model.pt` detects license plates across the image. RapidOCR immediately processes these crops to extract the text.
4. **Rider Association:** Riders are assigned to their respective motorcycles using bounding box overlap (Intersection over Union) and spatial heuristics (bottom-center point matching).
5. **Violation Logic:** The system checks the top 40% (head region) of each assigned rider to count helmet violations. If violations are present (helmetless riders or >2 riders), the system assigns the closest detected license plate to the motorcycle.

## 📊 Datasets

The custom models were trained using the following datasets (recommended for further analysis or extension):
- **Helmet Detection:** Safety Helmet Wearing Dataset (SHWD)
- **License Plates:** CCPD (Chinese City Parking Dataset), UFPR-ALPR
- **Riders/Vehicles:** Pre-trained on MS-COCO / BDD100K concepts.

## 📝 Notes

- **Offline Inference:** All models are loaded locally from the `models/` directory, requiring no internet connection for inference.
- **Customizability:** You can swap in your own custom detectors simply by replacing the corresponding `.pt` model files in the `models/` directory.
- **OCR Optimization:** RapidOCR ONNX models are warmed up during initialization to prevent execution provider conflicts with PyTorch.
