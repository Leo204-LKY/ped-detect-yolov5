#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include "detectAndDraw.cpp"
#include "resizeToFit.cpp"

using namespace std;
using namespace cv;

const string modelPath = "model.onnx";

int main(int argc, char** argv) {
	dnn::Net net = dnn::readNetFromONNX(modelPath);

	if (net.empty()) {
		cerr << "Failed to load the model from " << modelPath << endl;
		return -1;
	}

	net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(dnn::DNN_TARGET_CPU);

	string mode = "camera"; // Default mode is camera
	string path = "";

	if (argc >= 2) mode = argv[1];
	if (argc >= 3) path = argv[2];

	if (mode != "camera" && argc < 3) {
		cerr << "Please provide a file path." << endl;
	}

	if (mode == "camera" || mode == "video") {
		VideoCapture cap;
		if (mode == "camera") {
			cap.open(0);  // 0 for the default camera
		}
		else {
			cap.open(path);
		}

		if (!cap.isOpened()) {
			cerr << "Unable to open " << (mode == "camera" ? "camera." : path) << endl;
			return -1;
		}

		Mat frame;
		auto lastTick = chrono::steady_clock::now();
		float fps = 0.0f;

		while (true) {
			cap >> frame;
			if (frame.empty()) {
				cap.set(CAP_PROP_POS_FRAMES, 0);  // Return to 1st frame if the video ends
				continue;
			}

			Mat resizedFrame = resizeToFit(frame);

			// Record start time for FPS calculation
			auto startTick = chrono::steady_clock::now();

			// Person count
			int personCount = detectAndDraw(resizedFrame, net);

			// FPS calculation
			auto endTick = chrono::steady_clock::now();
			float sec = chrono::duration_cast<chrono::milliseconds>(endTick - startTick).count() / 1000.0f;
			fps = 1.0f / sec;

			// Display information on the frame
			string info = format("FPS  : %.2f", fps);
			string countInfo = "Count: " + to_string(personCount);
			string hint = "Press ESC to exit";

			putText(resizedFrame, info, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
			putText(resizedFrame, countInfo, Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
			putText(resizedFrame, hint, Point(10, 75), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

            imshow((mode == "camera" ? "Real-time Detection" : "Video Detection"), resizedFrame);
			if (waitKey(1) == 27) break; // Exit on 'ESC' key
		}
		cap.release();
	}
	else if (mode == "image") {
		Mat image = imread(path);
		if (image.empty()) {
			cerr << "Unable to open the image file: " << path << endl;
			return -1;
		}

		// Resize the image to fit the window
		Mat resizedImage = resizeToFit(image, 640, 640);

		int personCount = detectAndDraw(resizedImage, net);

		putText(resizedImage, "Count: " + to_string(personCount), Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
		putText(resizedImage, "Press any key to exit", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

		imshow("Image Detection", resizedImage);
		waitKey(0); // Wait for a key press to close the window
	}
	else {
		cerr << "Unknown mode: " << mode << ". Use 'camera', 'image <path>', or 'video <path>'." << endl;
		return -1;
	}
	destroyAllWindows();
	return 0;
}