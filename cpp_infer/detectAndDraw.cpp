#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

using namespace std;
using namespace cv;

const vector<string> classNames = { "person" };	// Types of objects to detect, persons only in this case
const string modelPath = "model.onnx";

/*
Process steps:
1. Load model (using OpenCV to read ONNX model)
2. Load image
3. Preprocess image (resize, normalize, convert to blob)
4. Inference (forward pass)
5. Post-process output (extract bounding boxes, apply NMS)
6. Visualize results (draw bounding boxes on the image)
*/

int main() {
	// Load the pre-trained model
	dnn::Net net = dnn::readNetFromONNX(modelPath);

	if (net.empty()) {
		cerr << "Failed to load the model from " << modelPath << endl;
		return -1;
	}

	cout << "Succcessfully load the model." << endl;

	// Set the preferable backend (OpenCV) and target (CPU)
	net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(dnn::DNN_TARGET_CPU);

	// Load an image
	Mat image = imread("image.jpg");

	if (image.empty()) {
		cerr << "Failed to load the image." << endl;
		return -1;
	}

	int inputWidth = 640;
	int inputHeight = 640;

	// Convert the image into blob format for model input
	// 1. Normalize the pixel values to [0, 1]
	// 2. Resize the image to the input size expected by the model (640x640)
	// 3. Swap the color channels from BGR to RGB
	Mat blob;
	dnn::blobFromImage(image, blob, 1.0 / 255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
	net.setInput(blob);

	// Forward inference
	vector<Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	// Print output details
	cout << "Output shape: " << outputs[0].size << endl;
	cout << "Output rows: " << outputs[0].rows << ", cols: " << outputs[0].cols << endl;

	// Process the outputs
	float confidenceThreshold = 0.25;
	float nmsThreshold = 0.45;

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	// Reshape the output to a 2D matrix
	Mat output = outputs[0];
	output = output.reshape(1, output.size[1]);

	float xFactor = image.cols / (float)inputWidth;
	float yFactor = image.rows / (float)inputHeight;

	// Extract the bounding box, confidence, and class
	// Iterate through the output rows
	for (int i = 0; i < output.rows; ++i) {
		float* data = (float*)output.ptr(i);
		float conf = data[4];
		float cls_score = data[5];
		float final_conf = conf * cls_score;

		if (final_conf > confidenceThreshold) {
			int centerX = static_cast<int>(data[0] * xFactor);
			int centerY = static_cast<int>(data[1] * yFactor);
			int width   = static_cast<int>(data[2] * xFactor);
			int height  = static_cast<int>(data[3] * yFactor);
			int left = centerX - width / 2;
			int top  = centerY - height / 2;

			classIds.push_back(0);	// "person" class
			confidences.push_back(final_conf);
			boxes.emplace_back(left, top, width, height);
		}
	}

	// Apply Non-Maximum Suppression
	// Used to remove overlapping bounding boxes
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

	// Draw the results
	for (int idx : indices) {
		Rect box = boxes[idx];
		rectangle(image, box, Scalar(0, 255, 0), 2);
		string label = format("%.2f", confidences[idx]);
		if (!classNames.empty()) {
			label = classNames[classIds[idx]] + ": " + label;
		}
		putText(image, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

		cout << "Detected: " << label << " at [" << box.x << ", " << box.y << ", " 
			 << box.width << ", " << box.height << "] with confidence: " 
			<< confidences[idx] << endl;
	}

	// Display the output
	imshow("Result", image);
	waitKey(0);
	return 0;
}