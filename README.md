## References
- https://blog.csdn.net/qq_65966646/article/details/149064056
- https://blog.csdn.net/guyuealian/article/details/128821763

## How to use
You can use the model in [Releases Page](https://github.com/Leo204-LKY/ped-detect-yolov5/releases) or train your own model with the following steps.

### Training
1. Download the VOC2012 Dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
2. Decompress it to `VOC2012`
3. Run the script to convert the dataset into YOLO type
   ```shell
   python utils/voc_to_yolo.py
   ```
    - Since this repository only detects `person`, the script will only convert images with person into YOLO type
4. Clone the YOLOv5 repository to `yolov5` folder
   ```shell
   git clone https://github.com/ultralytics/yolov5.git
   ```
5. Install PyTorch and other dependencies in `yolov5/requirements.txt`
6. Start training
   ```shell
   cd yolov5
   python train.py \
    --data ../voc2012.yaml \        # Config file of dataset
    --cfg models/yolov5n.yaml \     # Config file of model structure
    --weights yolov5n.pt \          # Load pre-trained weight file
    --batch-size 16 \               # Number of images used in each iteration
    --epochs 50 \                   # Total number of training rounds
    --device 0 \                    # Set up the device to be used (0 for GPU)
    --imgsz 480 \                   # Input image size
    --workers 4 \                   # Number of parallel threads when loading data
    --cache ram \                   # Cache images in memory (RAM) to speed up data loading.
    --noautoanchor \                # Turn off automatic anchor frame optimization
    --hyp data/hyps/hyp.scratch-low.yaml    # Specify hyperparameter file
   ```
   ```shell
   python train.py --data ../voc2012.yaml --cfg models/yolov5n.yaml --weights yolov5n.pt --batch-size 16 --epochs 50 --device 0 --imgsz 640 --workers 4 --cache ram --noautoanchor --hyp data/hyps/hyp.scratch-low.yaml
   ```
7. The output will be saved to `yolov5/runs/train/exp[x]`

### Convert to ONNX
```shell
python yolov5/export.py --weights "yolov5/runs/train/exp/weights/best.pt" --include onnx --simplify --opset 11
```
- Put the generated ONNX model in the same folder as `infer.py` (Python) or `cpp_infer` (C++), the script will automatically load it.

### Using the model
- `utils/py_infer.py` provides testing with Python. This program reads camera input and display the model output.
    - This is for testing the model only, the C++ version has better performace, more input formats, and more comments for studying.
- The C++ version of the inference script is in `cpp_infer` folder, use it for image, video or camera input.
    - Compile the C++ code with Visual Studio `.sln`
    - Commands:
        - Run the program with camera input, the default camera is 0.
            ```shell
            ./cpp_infer.exe [camera]
            ```
        - Run the program with image input, replace `<image_path>` with the path to the image file.
            ```shell
            ./cpp_infer.exe image <image_path>
            ```
        - Run the program with video input, replace `<video_path>` with the path to the video file.
            ```shell
            ./cpp_infer.exe video <video_path>
            ```