<<<<<<< HEAD
# YOLOv8 Object Detection App

A comprehensive Streamlit-based object detection application using YOLOv8 models from Ultralytics. This app provides an intuitive web interface for uploading images, detecting objects, and visualizing results with customizable settings.

## Features

- 🎯 **Multi-format Model Support**: Load both PyTorch (.pt) and ONNX (.onnx) models
- 🔄 **ONNX Conversion**: Convert PyTorch models to ONNX format within the app
- ⚙️ **Customizable Detection**: Adjust confidence and IoU thresholds
- 🔍 **Class Filtering**: Filter detections by specific object classes
- 🎨 **Visual Annotations**: Bounding boxes with class labels and confidence scores
- 📊 **Detection Statistics**: Real-time statistics and class-wise counts
- 📥 **Result Download**: Download annotated images with detections
- 🔄 **Data Augmentation**: Enable augmentation during inference
- 🎨 **Beautiful UI**: Modern, responsive interface with custom styling
- 🚀 **Session Management**: Persistent state across interactions

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add YOLOv8 models**:
   - Place your `.pt` or `.onnx` model files in the project directory
   - The app will automatically detect `yolo11n.pt` as the default model
   - Supported formats: `.pt` (PyTorch) and `.onnx` (ONNX)

## Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**:
   - Open your browser and go to `http://localhost:8501`
   - The app will automatically load available models

3. **Using the app**:
   - **Model Selection**: Choose from available models in the sidebar
   - **ONNX Conversion**: Convert PyTorch models to ONNX format
   - **Upload Image**: Select an image file to analyze
   - **Adjust Settings**: Modify confidence, IoU thresholds, and class filters
   - **Run Detection**: Click "Detect Objects" to process the image
   - **View Results**: See annotated image with bounding boxes and labels
   - **Download Results**: Save the detection result image

## Model Requirements

### Supported Models
- **YOLOv8**: All variants (nano, small, medium, large, xlarge)
- **Formats**: PyTorch (.pt) and ONNX (.onnx)
- **Classes**: COCO dataset classes (80 classes)

### Model Files
- Place model files in the project root directory
- Default model: `yolo11n.pt`
- The app automatically scans for `.pt` and `.onnx` files

## Configuration Options

### Detection Settings
- **Confidence Threshold**: Minimum confidence score (0.1 - 1.0)
- **IoU Threshold**: Non-maximum suppression threshold (0.1 - 1.0)
- **Augmentation**: Enable/disable data augmentation during inference

### Class Filtering
- **Multi-select**: Choose specific classes to detect
- **All Classes**: Leave empty to detect all available classes
- **Common Classes**: Pre-populated with COCO dataset classes

### Visual Settings
- **Bounding Boxes**: Red for "person", blue for other classes
- **Labels**: Class name and confidence score
- **Statistics**: Real-time detection counts and class breakdown

## File Structure

```
yolov8_streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── yolo11n.pt         # Default YOLOv8 nano model
├── yolo11n.onnx       # ONNX version (if converted)
└── venv/              # Virtual environment (if used)
```

## Dependencies

- **streamlit**: Web application framework
- **ultralytics**: YOLOv8 model library
- **opencv-python**: Computer vision and image processing
- **Pillow**: Image handling
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities
- **onnx**: ONNX format support
- **onnxruntime**: ONNX model inference

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure model files are in the correct directory
   - Check file permissions
   - Verify model format compatibility

2. **Memory Issues**:
   - Use smaller models (nano/small) for limited resources
   - Reduce image resolution
   - Close other applications

3. **OpenMP Compatibility**:
   - The app includes automatic OpenMP fix
   - Set `KMP_DUPLICATE_LIB_OK=TRUE` environment variable

4. **CUDA/GPU Issues**:
   - Install appropriate CUDA drivers
   - Use CPU-only models if GPU unavailable
   - Check PyTorch CUDA compatibility

### Performance Tips

- **Faster Inference**: Use ONNX models for better performance
- **Memory Optimization**: Use smaller model variants
- **Batch Processing**: Process multiple images sequentially
- **Caching**: Models are cached for faster subsequent loads

## Advanced Features

### ONNX Conversion
- Convert PyTorch models to ONNX format
- Automatic model optimization
- Cross-platform compatibility
- Better inference performance

### Session State Management
- Persistent model selections
- Automatic model discovery
- State preservation across interactions

### Error Handling
- Graceful error messages
- Model validation
- Input validation
- Fallback mechanisms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **Ultralytics**: YOLOv8 model library
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify model file compatibility

---

**Built with ❤️ using Streamlit and YOLOv8** 
=======
# Object-Detection-using-Yollo-model
>>>>>>> fa0647cbf0e64f6bd300a9f2da702c0c41e52b7c
