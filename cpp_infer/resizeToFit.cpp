#include <opencv2/opencv.hpp>

using namespace cv;

static Mat resizeToFit(const Mat& input, int maxWidth = 1280, int maxHeight = 1280) {
	Mat output;
	float scale = min((float)maxWidth / input.cols, (float)maxHeight / input.rows);
	if (scale < 1.0f) {
		resize(input, output, Size(), scale, scale);
	}
	else {
		output = input.clone();
	}
	return output;
}