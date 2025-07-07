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

	// Calculate the scaling factors for bounding box coordinates
	// Used to convert the normalized coordinates back to the original frame size
	float xFactor = frame.cols / (float)inputWidth;
	float yFactor = frame.rows / (float)inputHeight;

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	// Extract the bounding box, confidence, and class
	// Iterate through the output rows
	// Each row contains: [centerX, centerY, width, height, confidence, classScore1, classScore2, ...]
	// We only detect "person" class (class 0), so only classScore1 is relevant
	for (int i = 0; i < output.rows; ++i) {
		// Get the head address pointer of the current row i, and convert it to a float array
		float* data = (float*)output.ptr(i);
		float conf = data[4];					// Confidence score for the detection
		float cls_score = data[5];				// Class score for the "person" class (index 0)
		float final_conf = conf * cls_score;	// Final confidence score: confidence * class score

		// Only consider detections with confidence above the threshold
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