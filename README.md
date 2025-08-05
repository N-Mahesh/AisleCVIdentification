# Aisle Product Camera Identification

An AI-powered system that automatically identifies and catalogs products in grocery store aisles using computer vision and natural language processing. This project combines YOLOv10 object detection with Claude AI vision capabilities to provide comprehensive product recognition from aisle photographs.

## 🌟 Features

- **Object Detection**: Uses YOLOv10 model trained on SKU-110K dataset for accurate product detection in grocery aisles
- **Product Identification**: Leverages Claude AI vision to identify specific products from cropped images
- **Parallel Processing**: Efficiently processes multiple products simultaneously using multiprocessing
- **Web Interface**: Flask-based web application with real-time progress tracking via ngrok tunneling
- **Batch Processing**: Handles multiple product identifications from a single aisle image

## 🔧 Technical Stack

- **Machine Learning**: YOLOv10, Ultralytics
- **AI Vision**: Anthropic Claude 3.5 Sonnet
- **Backend**: Python, Flask
- **Image Processing**: PIL, OpenCV, Matplotlib
- **Web Framework**: Flask with Bulma CSS
- **Deployment**: ngrok for public access
- **Data Processing**: Pickle for serialization, concurrent.futures for parallel processing

## 📋 Prerequisites

### Required Packages
```bash
pip install ultralytics
pip install anthropic
pip install flask-ngrok
pip install pyngrok
pip install tqdm
pip install pillow
pip install matplotlib
```

### API Keys Required
- **Anthropic Claude API Key**: For product identification
- **ngrok Auth Token**: For web interface tunneling

## 🚀 Quick Start

### 1. Model Training
The system uses a YOLOv10 model trained on the SKU-110K dataset:

```python
from ultralytics import YOLO

# Create and train the model
model = YOLO("yolov10m.pt")
results = model.train(data="SKU-110K.yaml", epochs=10, imgsz=640)
```

### 2. Product Detection
Load an aisle image and detect products:

```python
# Load trained model
model = YOLO("path/to/trained/model.pt")

# Run inference on aisle image
results = model("path/to/aisle/image.jpg")
```

### 3. Product Identification
The system automatically crops detected products and identifies them using Claude AI:

```python
# Process croppings with parallel identification
identify_croppings_parallel(croppings_folder, output_file)
```

### 4. Web Interface
Launch the Flask web application:

```python
# Set up ngrok tunnel
public_url = ngrok.connect(5000)
app.run(port=5000)
```

## 📁 Project Structure

```
AisleCVIdentification/
├── AisleProductCameraIdentification.ipynb  # Main notebook
├── README.md                               # This file
├── models/
│   └── YOLO_model_e10_new.pt             # Trained YOLOv10 model
├── data/
│   ├── cropped_images/                    # Product croppings
│   └── identifications.pkl               # Serialized results
└── utils/
    └── SKU-110K.yaml                      # Dataset configuration
```

## 🔄 Workflow

1. **Image Input**: Upload an aisle photograph through the web interface
2. **Object Detection**: YOLOv10 identifies and crops individual products
3. **Product Recognition**: Claude AI analyzes each cropping to identify specific products
4. **Result Compilation**: System generates keywords and product names for each item
5. **Output Delivery**: Results delivered via web interface with real-time progress tracking

## 🎯 Model Performance

The YOLOv10m model achieves:
- **mAP50**: 0.907 (90.7% accuracy at 50% IoU threshold)
- **mAP50-95**: 0.589 (58.9% accuracy across IoU thresholds)
- **Precision**: 0.902
- **Recall**: 0.838

Training was performed on 588 images with 90,968 product instances.

## 🌐 Web Interface Features

- **Upload Interface**: Simple file upload for aisle images
- **Real-time Progress**: Live updates during processing
- **Responsive Design**: Bulma CSS framework for modern UI
- **Public Access**: ngrok tunneling for external accessibility

## 📊 Output Format

The system provides structured output for each identified product:

```json
{
  "keywords": ["brand name", "product type", "color", "packaging"],
  "product_name": "Specific Product Name"
}
```

## 🔒 Security & Privacy

- API keys are externalized and not stored in code
- Temporary file processing with automatic cleanup
- Secure tunneling through ngrok authentication

## 🚧 Development Notes

### Training Environment
- Optimized for Google Colab with T4 GPU
- Training time: ~3 hours for 10 epochs
- Memory usage: ~10.9GB GPU memory

### Performance Optimization
- Parallel processing for multiple product identification
- Efficient memory management with base64 encoding
- Progress tracking for long-running operations

## 📝 Usage Examples

### Basic Product Detection
```python
# Load image and get croppings
get_croppings("path/to/aisle/image.jpg")

# Identify all products
identify_croppings_parallel(croppings_folder, output_file)

# Load and display results
with open(output_file, 'rb') as f:
    results = pickle.load(f)
    for product in results:
        print(product['response_data'])
```

### Web Application
1. Start the application
2. Navigate to the provided ngrok URL
3. Upload an aisle image
4. Monitor real-time processing progress
5. Receive JSON output with product identifications

## 🤝 Contributing

This project is part of academic research in computer vision and AI applications in retail environments. Contributions focusing on:
- Model accuracy improvements
- Processing speed optimization
- Additional AI vision model integration
- Enhanced web interface features

## 📄 License

**All Rights Reserved**

This project and all associated code, documentation, and materials are the exclusive property of the creator. No part of this project may be used, copied, modified, distributed, or otherwise utilized without explicit written permission from the creator.

**Usage Restrictions:**
- Commercial use is strictly prohibited without written authorization
- Academic or research use requires prior approval and proper attribution
- Redistribution in any form is not permitted without express consent
- Modification or derivative works require explicit permission

For permission requests or licensing inquiries, please contact the project creator directly.

**Third-Party Services:** Users must ensure compliance with respective API terms of service (Anthropic Claude, ngrok) when using this system.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv10 implementation
- **SKU-110K Dataset** for training data
- **Anthropic** for Claude AI vision capabilities

---

*For questions or support, please refer to the notebook documentation or create an issue in the repository.*
