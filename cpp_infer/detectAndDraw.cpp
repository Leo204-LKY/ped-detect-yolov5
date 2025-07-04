#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

using namespace std;
using namespace cv;

const vector<string> classNames = { "person" };	// Types of objects to detect, persons only in this case

/*
Process steps:
1. Preprocess frame (resize, normalize, convert to blob)
2. Inference (forward pass)
3. Post-process output (extract bounding boxes, apply NMS)
4. Visualize results (draw bounding boxes on the frame)
*/

static int detectAndDraw(
	Mat& frame,
	dnn::Net& net,
	float confThreshold = 0.25,
	float nmsThreshold = 0.25
) {
	int inputWidth = 640;
	int inputHeight = 640;

	// Convert the frame into blob format for model input
	// 1. Normalize the pixel values to [0, 1]
	// 2. Resize the frame to the input size expected by the model (640x640)
	// 3. Swap the color channels from BGR to RGB
	Mat blob;
	dnn::blobFromImage(frame, blob, 1.0 / 255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
	net.setInput(blob);

	// Forward inference
	vector<Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	// Process the outputs
	// Reshape the output to a 2D matrix
	Mat output = outputs[0];
	output = output.reshape(1, output.size[1]);

	float xFactor = frame.cols / (float)inputWidth;
	float yFactor = frame.rows / (float)inputHeight;

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	// Extract the bounding box, confidence, and class
	// Iterate through the output rows
	for (int i = 0; i < output.rows; ++i) {
		float* data = (float*)output.ptr(i);
		float conf = data[4];
		float cls_score = data[5];
		float final_conf = conf * cls_score;

		if (final_conf > confThreshold) {
			int centerX = static_cast<int>(data[0] * xFactor);
			int centerY = static_cast<int>(data[1] * yFactor);
			int width = static_cast<int>(data[2] * xFactor);
			int height = static_cast<int>(data[3] * yFactor);
			int left = centerX - width / 2;
			int top = centerY - height / 2;

			classIds.push_back(0);	// "person" class
			confidences.push_back(final_conf);
			boxes.emplace_back(left, top, width, height);
		}
	}

	// Apply Non-Maximum Suppression
	// Used to remove overlapping bounding boxes
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// Draw the results
	for (int idx : indices) {
		Rect box = boxes[idx];
		rectangle(frame, box, Scalar(0, 0, 255), 2);
		string label = format("%.2f", confidences[idx]);
		if (!classNames.empty()) {
			label = classNames[classIds[idx]] + ": " + label;
		}
		putText(frame, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
	}

	return (int)indices.size();
}