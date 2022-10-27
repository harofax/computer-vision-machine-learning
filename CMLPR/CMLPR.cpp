// CMLPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <baseapi.h>
#include <allheaders.h>

using namespace cv;
using namespace std;
Mat RGB2Grey(Mat RGB)
{
	Mat grey = Mat::zeros(RGB.size(), CV_8UC1);

	for (int i = 0; i < RGB.rows; i++)
	{
		for (int j = 0; j < RGB.cols * 3; j += 3)
		{
			grey.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
		}
	}

	return grey;
}

Mat Grey2Binary(Mat grey, int threshold)
{
	Mat bin = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			if (grey.at<uchar>(i, j) > threshold)
			{
				bin.at<uchar>(i, j) = 255;
			}
		}
	}

	return bin;
}

Mat Grey2Inverted(Mat grey)
{
	Mat invert = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			invert.at<uchar>(i, j) = 255 - grey.at<uchar>(i, j);
		}
	}

	return invert;
}

Mat Step(Mat grey, int threshLower, int threshUpper)
{
	Mat output = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			if (grey.at<uchar>(i, j) >= threshLower &&
				grey.at<uchar>(i, j) <= threshUpper)
			{
				output.at<uchar>(i, j) = 255;
			}
		}
	}

	return output;
}

Mat Downscale(Mat RGB, int scale)
{
	Mat downscaled = Mat::zeros(RGB.size() / scale, RGB.type());

	for (int i = 0; i < downscaled.rows; i++)
	{
		for (int j = 0; j < downscaled.cols * 3; j += 3)
		{
			//downscaled.at<uchar>(i, j) = RGB.at<uchar>(i * scale, j * scale);
			downscaled.at<uchar>(i, j)		= RGB.at<uchar>(i * scale, j * scale);
			downscaled.at<uchar>(i, j + 1)	= RGB.at<uchar>(i * scale, (j * scale + 1));
			downscaled.at<uchar>(i, j + 2)	= RGB.at<uchar>(i * scale, (j * scale + 2));
		}
	}

	return downscaled;
}

Mat UpScalePoorly(Mat RGB, int scale)
{
	Mat upscaled = Mat::zeros(RGB.size() * scale, RGB.type());

	for (int i = 0; i < upscaled.rows; i++)
	{
		for (int j = 0; j < upscaled.cols * 3; j += 3)
		{
			upscaled.at<uchar>(i, j)		= RGB.at<uchar>(i / scale, j / scale);
			upscaled.at<uchar>(i, j + 1)	= RGB.at<uchar>(i / scale, (j / scale + 1));
			upscaled.at<uchar>(i, j + 2)	= RGB.at<uchar>(i / scale, (j / scale + 2));
			//upscaled.at<uchar>(i, j) = RGB.at<uchar>(i / scale, j / scale);
		}
	}

	return upscaled;
}

Mat BetterScale(Mat RGB, int scale)
{
	Mat betterscaled = Mat::zeros(RGB.size() * scale, RGB.type());

	int sample_i = 0;
	int sample_j = 0;

	for (int i = 0; i < betterscaled.rows; i++)
	{
		for (int j = 0; j < betterscaled.cols; j += 3)
		{
			sample_i = (int)std::max(i * 1.0f / scale, (float)betterscaled.rows - 1);
			sample_j = (int)std::max(j * 1.0f / scale, (float)betterscaled.cols - 1);


			betterscaled.at<uchar>(i, j) = RGB.at<uchar>(sample_i, sample_j);
			betterscaled.at<uchar>(i, j + 1) = RGB.at<uchar>(sample_i, sample_j + 1);
			betterscaled.at<uchar>(i, j + 2) = RGB.at<uchar>(sample_i, sample_j + 2);
		}
	}

	return betterscaled;
}

Mat NaturalSelection(Mat RGB)
{
	Mat evolution = Mat::zeros(RGB.size(), RGB.type());

	for (int i = 0; i < RGB.rows; i++)
	{
		for (int j = 0; j < RGB.cols * 3; j += 3)
		{
			auto b = RGB.at<uchar>(i, j);
			auto g = RGB.at<uchar>(i, j + 1);
			auto r = RGB.at<uchar>(i, j + 2);

			auto strongest = std::max({ r, g, b });

			if (strongest == r)
			{
				evolution.at<uchar>(i, j) = 0;
				evolution.at<uchar>(i, j + 1) = 0;
				evolution.at<uchar>(i, j + 2) = r;
			}
			else if (strongest == g)
			{
				evolution.at<uchar>(i, j) = 0;
				evolution.at<uchar>(i, j + 1) = g;
				evolution.at<uchar>(i, j + 2) = 0;
			}
			else if (strongest == b)
			{
				evolution.at<uchar>(i, j) = b;
				evolution.at<uchar>(i, j + 1) = 0;
				evolution.at<uchar>(i, j + 2) = 0;
			} else
			{
				evolution.at<uchar>(i, j) = 255;
				evolution.at<uchar>(i, j + 1) = 0;
				evolution.at<uchar>(i, j + 2) = 255;
			}

			
		}
	}

	return evolution;
}

float Average(Mat grey)
{
	int sum = 0;

	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			sum += grey.at<uchar>(i, j);
		}
	}
	return sum / (grey.rows * grey.cols);
}

Mat Blur3x3(Mat grey)
{
	Mat blurred = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 1; i < grey.rows - 1; i++)
	{
		for (int j = 1; j < grey.cols - 1; j++)
		{
			int average = 0;
			for (int n_i = -1; n_i <= 1; n_i++)
			{
				for (int n_j = -1; n_j <= 1; n_j++)
				{
					average += grey.at<uchar>(i + n_i, j + n_j);
				}
			}
			blurred.at<uchar>(i, j) = (average / 9.0);
		}
	}

	return blurred;
}

Mat Blur(Mat grey, int neighborSize)
{
	Mat blurredImage = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0 + neighborSize; i < grey.rows - neighborSize; i++)
	{
		for (int j = 0 + neighborSize; j < grey.cols - neighborSize; j++)
		{
			int sum = 0;
			int neighbors = 0;

			for (int x = -neighborSize; x <= neighborSize; x++)
			{
				for (int y = -neighborSize; y <= neighborSize; y++)
				{
					if (x != 0 && y != 0)
					{
						sum += grey.at<uchar>(i + x, j + y);
						neighbors++;
					}
				}
			}

			blurredImage.at<uchar>(i, j) = sum / neighbors;
		}
	}
	return blurredImage;
}

Mat ErosionOpt(Mat Edge, int windowsize) {

	Mat ErodedImg = Mat::zeros(Edge.size(), CV_8UC1);

	for (int i = windowsize; i < Edge.rows - windowsize; i++) 
	{
		for (int j = windowsize; j < Edge.cols - windowsize; j++) 
		{
			ErodedImg.at<uchar>(i, j) = Edge.at<uchar>(i, j);

			for (int p = -windowsize; p <= windowsize; p++) 
			{
				for (int q = -windowsize; q <= windowsize; q++) 
				{
					if (Edge.at<uchar>(i + p, j + q) == 0) 
					{
						ErodedImg.at<uchar>(i, j) = 0;
					}
				}
			}
		}
	}
	return ErodedImg;
}

Mat MaxNxN(Mat grey, int N)
{
	Mat maxxed = Mat::zeros(grey.size(), CV_8UC1);

	int neighbour_range = (N - 1) / 2;

	for (int i = 1; i < grey.rows - 1; i++)
	{
		for (int j = 1; j < grey.cols - 1; j++)
		{
			int max = 0;
			for (int neighbour_i = -neighbour_range; neighbour_i <= neighbour_range; neighbour_i++)
			{
				for (int neighbour_j = -neighbour_range; neighbour_j <= neighbour_range; neighbour_j++)
				{
					auto current = grey.at<uchar>(i + neighbour_i, j + neighbour_j);
					if (current > max)
					{
						max = current;
					}
				}
			}
			maxxed.at<uchar>(i, j) = max;
		}
	}

	return maxxed;
}

Mat MinNxN(Mat grey, int N)
{
	Mat minned = Mat::zeros(grey.size(), CV_8UC1);

	int neighbour_range = (N - 1) / 2;

	for (int i = 1; i < grey.rows - 1; i++)
	{
		for (int j = 1; j < grey.cols - 1; j++)
		{
			int min = 255;
			for (int neighbour_i = -neighbour_range; neighbour_i <= neighbour_range; neighbour_i++)
			{
				for (int neighbour_j = -neighbour_range; neighbour_j <= neighbour_range; neighbour_j++)
				{
					auto current = grey.at<uchar>(i + neighbour_i, j + neighbour_j);
					if (current < 255)
					{
						min = current;
					}
				}
			}
			minned.at<uchar>(i, j) = min;
		}
	}

	return minned;
}

Mat Min(Mat grey, int neighborSize)
{
	Mat minimizedImage = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0 + neighborSize; i < grey.rows - neighborSize; i++)
	{
		for (int j = 0 + neighborSize; j < grey.cols - neighborSize; j++)
		{
			int currentLowest = 255;

			for (int x = -neighborSize; x <= neighborSize; x++)
			{
				for (int y = -neighborSize; y <= neighborSize; y++)
				{
					if (x != 0 && y != 0 && grey.at<uchar>(i + x, j + y) < currentLowest)
					{
						currentLowest = grey.at<uchar>(i + x, j + y);
					}
				}
			}

			minimizedImage.at<uchar>(i, j) = currentLowest;
		}
	}
	return minimizedImage;
}

Mat Max(Mat grey, int neighborSize)
{
	Mat maxedImage = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0 + neighborSize; i < grey.rows - neighborSize; i++)
	{
		for (int j = 0 + neighborSize; j < grey.cols - neighborSize; j++)
		{
			int currentHighest = -1;

			for (int x = -neighborSize; x <= neighborSize; x++)
			{
				for (int y = -neighborSize; y <= neighborSize; y++)
				{
					if (x != 0 && y != 0 && grey.at<uchar>(i + x, j + y) > currentHighest)
					{
						currentHighest = grey.at<uchar>(i + x, j + y);
					}
				}
			}

			maxedImage.at<uchar>(i, j) = currentHighest;
		}
	}
	return maxedImage;
}

Mat BlurNxN(Mat grey, int N)
{
	Mat blurred = Mat::zeros(grey.size(), CV_8UC1);

	const int neighbour_range = (N - 1) / 2;

	for (int i = 1; i < grey.rows - 1; i++)
	{
		for (int j = 1; j < grey.cols - 1; j++)
		{
			int total = 0;
			int count = 0;
			for (int neighbour_i = -neighbour_range; neighbour_i <= neighbour_range; neighbour_i++)
			{
				for (int neighbour_j = -neighbour_range; neighbour_j <= neighbour_range; neighbour_j++)
				{
					count++;
					total += grey.at<uchar>(i + neighbour_i, j + neighbour_j);
				}
			}
			blurred.at<uchar>(i, j) = (total / count);
		}
	}

	return blurred;
}

Mat Edge(Mat grey, int thresh)
{
	Mat edge = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 1; i < grey.rows - 1; i++)
	{
		for (int j = 1; j < grey.cols - 1; j++)
		{
			int avg_L = (grey.at<uchar>(i - 1, j - 1) + grey.at<uchar>(i, j - 1) + grey.at<uchar>(i + 1, j - 1)) / 3;
			int avg_R = (grey.at<uchar>(i - 1, j + 1) + grey.at<uchar>(i, j + 1) + grey.at<uchar>(i + 1, j + 1)) / 3;

			if (abs(avg_L - avg_R) > thresh)
			{
				edge.at<uchar>(i, j) = 255;
			}
		}
	}

	return edge;
}

Mat Dilation(Mat binary_img, int border = 1)
{
	Mat dilated = Mat::zeros(binary_img.size(), CV_8UC1);

	for (int i = border; i < binary_img.rows - border; i++)
	{
		for (int j = border; j < binary_img.cols - border; j++)
		{
			// if white, skip
			if (binary_img.at<uchar>(i, j) == 255)
			{
				dilated.at<uchar>(i, j) = 255;
				continue;
			}

			for (int ii = -border; ii <= border; ii++)
			{
				for (int jj = -border; jj <= border; jj++)
				{
					int value = binary_img.at<uchar>(i + ii, j + jj);

					// if neighbours are white, make white
					if (value == 255)
					{
						dilated.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}
	return dilated;
}

Mat DetectAndCrop(Mat grey, int range)
{
	Mat croppedImage = Mat::zeros(grey.size(), CV_8UC1);
	int counter = 0;

	for (int i = 1; i < grey.rows - 1; i++)
	{
		for (int j = 1; j < grey.cols - 1; j++)
		{
			if (grey.at<uchar>(i, j) == 255)
			{
				for (int x = 1; x < range; x++)
				{
					if (grey.at<uchar>(i + x, j) != 255)
					{
						break;
					}

					counter++;
				}

				if (counter >= range)
				{
					// put cropping code here to crop out a rectangle starting from i,j as the top left corner
				}
			}

		}
	}
	return croppedImage;
}


Mat EqualizeHistogram(Mat grey)
{
	Mat equalizedImage = Mat::zeros(grey.size(), CV_8UC1);

	int count[256] = { 0 };
	float probability[256] = { 0.0 };
	float accumulatedProbability[256] = { 0.0 };
	int newValue[256] = { 0 };

	// incrementing count
	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			count[grey.at<uchar>(i, j)]++;
		}
	}

	// probability
	for (int i = 0; i < 256; i++)
	{
		probability[i] = (float)count[i] / (float)(grey.rows * grey.cols);
	}

	// accumulated probability
	accumulatedProbability[0] = probability[0];
	for (int i = 1; i < 256; i++)
	{
		accumulatedProbability[i] = probability[i] + accumulatedProbability[i - 1];
	}

	// get new value for pixels
	for (int i = 0; i < 256; i++)
	{
		newValue[i] = 255 * accumulatedProbability[i];
	}

	// set the values in the equalized image to be the new value for those pixels
	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			equalizedImage.at<uchar>(i, j) = newValue[grey.at<uchar>(i, j)];
		}
	}

	return equalizedImage;
}

int OTSU(Mat grey)
{
	// incrementing count
	int count[256] = { 0 };
	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			count[grey.at<uchar>(i, j)]++;
		}
	}

	// probability
	float probability[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
	{
		probability[i] = (float)count[i] / (float)(grey.rows * grey.cols);
	}

	// accumulated probability
	float theta[256] = { 0.0 };
	theta[0] = probability[0];
	for (int i = 1; i < 256; i++)
	{
		theta[i] = probability[i] + theta[i - 1];
	}

	float µ[256] = { 0.0 };
	for (int i = 1; i < 256; i++)
	{
		µ[i] = i * probability[i] + µ[i - 1];
	}

	float sigma[256] = { 0.0 };
	for (int i = 1; i < 256; i++)
	{
		sigma[i] = pow(µ[255] * theta[i] - µ[i], 2) / (theta[i] * (1 - theta[i]));
	}

	int index = 0;
	float maxVal = 0;
	for (int i = 0; i < 256; i++)
	{
		if (sigma[i] > maxVal)
		{
			maxVal = sigma[i];
			index = i;
		}
	}

	return index + 30;
}

int main()
{
	Mat img = imread("cars\\1_6.jpg");
	Mat grey = RGB2Grey(img);
	//imshow("Grayscale", grey);
	Mat binary_img = Grey2Binary(grey, 128);
	Mat inv_img = Grey2Inverted(grey);
	Mat stepped = Step(grey, 80, 140);

	Mat maxxed = Max(grey, 2);
	Mat minned = Min(grey, 2);

	//imshow("maxd", maxxed);
	//imshow("minned", minned);

	Mat histogrammed = EqualizeHistogram(grey);
	Mat blurred = BlurNxN(histogrammed, 3);

	//imshow("histo", histogrammed);
	//imshow("blurd histo", blurred);

	float threshold = Average(blurred);

	std::cout << threshold << std::endl;

	float multiplier = (1.0f * threshold) / 255.0f;

	int edge_threshold = 50;

	Mat edged = Edge(blurred, (int)(multiplier * edge_threshold));
	std::cout << (int)(multiplier * edge_threshold) << std::endl;

	Mat eroded = ErosionOpt(edged, 1);
	Mat dilated = Dilation(edged, 4);

	vector <vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(dilated, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	Mat dst = Mat::zeros(grey.size(), CV_8UC3);

	if (!contours.empty())
	{
		for (size_t i = 0; i < contours.size(); i++)
		{
			Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, contours, i, color, -1, 8, hierarchy);
		}
	}

	Mat plate;
	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);

	int numConts = contours.size();

	cout << "num conts start: " << numConts << endl;

	for (int i = 0; i < contours.size(); i++)
	{
		rect = boundingRect(contours[i]);

		float ratio = ((float)rect.width / (float)rect.height);

		if (rect.width < 65
			|| rect.width > 180
			|| rect.height > 100
			|| rect.x <= (grey.rows * 0.1f)
			|| rect.x >= (grey.rows * 0.9f)
			|| rect.y >= (grey.cols * 0.9f)
			|| rect.y <= (grey.cols * 0.1f)
			|| ratio < 1.5f)
		{
			drawContours(dilated, contours, i, black, -1, 8, hierarchy);
			numConts--;
		}
		else
		{
			plate = grey(rect);
		}
	}

	cout << "num conts end: " << numConts << endl;

	if (plate.rows != 0 && plate.cols != 0)
	{
		imshow("Plate", plate);
	}

	int OTSUThreshold = OTSU(plate);
	Mat binarizedPlate = Grey2Binary(plate, OTSUThreshold);

	Mat binaryPlateCopy = binarizedPlate.clone();
	vector <vector<Point>> contours1;
	vector<Vec4i> hierarchy1;

	if (binarizedPlate.rows != 0 && binarizedPlate.cols != 0)
	{
		imshow("Binarized Plate", binarizedPlate);
	}

	findContours(binarizedPlate, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	Mat dst1 = Mat::zeros(grey.size(), CV_8UC3);

	if (!contours1.empty())
	{
		for (size_t i = 0; i < contours1.size(); i++)
		{
			Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst1, contours1, i, color, -1, 8, hierarchy1);
		}
	}

	imshow("Segmented plate", dst1);

	Mat character;
	vector<Mat> letters;
	for (int i = 0; i < contours1.size(); i++)
	{
		rect = boundingRect(contours1[i]);

		if (rect.height < 5)
		{
			drawContours(dilated, contours, i, black, -1, 8, hierarchy);
		}
		else
		{
			character = binarizedPlate(rect);
			if (character.rows != 0 && character.cols != 0)
			{
				letters.push_back(character);
			}

			//imshow("Character", character);
			//waitKey();
		}
	}

	imshow("Segmented image", dst);

	tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

	api->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK);
	api->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

	if (api->Init("Q:\\gamedev\\programming\\CPP\\CMLPR\\", "eng"))
	{
		std::cout << "Could not initialize Tesseract!" << std::endl;
		exit(1);
	}


	if (!letters.empty())
	{
		for (int i = 0; i < letters.size(); i++)
		{
			resize(letters[i], letters[i], Size(letters[i].size().width * 4, letters[i].size().height * 4), 0, 0, INTER_LINEAR);

			letters[i] = Blur(letters[i], 3);

			letters[i] = Step(letters[i], 240, 255);

			api->SetImage(letters[i].data, letters[i].size().width, letters[i].size().height, letters[i].channels(), letters[i].step1());
			const char* letter = api->GetUTF8Text();
			std::cout << letter;
		}
	}
	
    waitKey();
}