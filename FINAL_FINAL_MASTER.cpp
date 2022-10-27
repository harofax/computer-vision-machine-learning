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

Mat Grey2Binary(Mat Grey, int threshold)
{
	Mat bin = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			if (Grey.at<uchar>(i, j) > threshold)
				bin.at<uchar>(i, j) = 255;

		}
	}

	return bin;
}

Mat Inversion(Mat Grey)
{
	Mat invertedImg = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			invertedImg.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);

		}
	}

	return invertedImg;
}

Mat Step(Mat Grey, int th1, int th2)
{
	Mat output = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			if (Grey.at<uchar>(i, j) >= th1 && Grey.at<uchar>(i, j) <= th2)
				output.at<uchar>(i, j) = 255;

		}
	}

	return output;
}

Mat Avg(Mat Grey, int neighbirSize)
{
	Mat AvgImg = Mat::zeros(Grey.size(), CV_8UC1);
	int totalPix = pow(2 * neighbirSize + 1, 2);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int sum = 0;
			int count = 0;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					count++;
					sum += Grey.at<uchar>(i + ii, j + jj);
				}
			}
			AvgImg.at<uchar>(i, j) = sum / count;
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;

		}
	}

	return AvgImg;
}

Mat Max(Mat Grey, int neighbirSize)
{
	Mat MaxImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int Defval = -1;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{

					if (Grey.at<uchar>(i + ii, j + jj) > Defval)
						Defval = Grey.at<uchar>(i + ii, j + jj);
				}
			}
			MaxImg.at<uchar>(i, j) = Defval;
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;

		}
	}

	return MaxImg;
}

Mat Min(Mat Grey, int neighbirSize)
{
	Mat MinImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int Defval = 255;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{

					if (Grey.at<uchar>(i + ii, j + jj) < Defval)
						Defval = Grey.at<uchar>(i + ii, j + jj);
				}
			}
			MinImg.at<uchar>(i, j) = Defval;
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;

		}
	}

	return MinImg;
}

Mat Edge(Mat Grey, int th)
{
	Mat EdgeImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++)
	{
		for (int j = 1; j < Grey.cols - 1; j++)
		{
			int AvgL = (Grey.at<uchar>(i - 1, j - 1) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i + 1, j - 1)) / 3;
			int AvgR = (Grey.at<uchar>(i - 1, j + 1) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1)) / 3;
			if (abs(AvgL - AvgR) > th)
				EdgeImg.at<uchar>(i, j) = 255;


		}
	}

	return EdgeImg;


}

Mat DilationOpt(Mat Edge, int windowsize)
{
	Mat dilatedImg = Mat::zeros(Edge.size(), CV_8UC1);
	for (int i = windowsize; i < Edge.rows - windowsize; i++)
	{
		for (int j = windowsize; j < Edge.cols - windowsize; j++)
		{
			for (int ii = -windowsize; ii <= windowsize; ii++)
			{
				for (int jj = -windowsize; jj <= windowsize; jj++)
				{
					if (Edge.at<uchar>(i + ii, j + jj) == 255)
					{
						dilatedImg.at<uchar>(i, j) = 255;
						break;
					}
					break;
				}
			}
		}
	}
	return dilatedImg;
}

Mat ErosionOpt(Mat Edge, int windowsize)
{
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

Mat Dilation(Mat EdgeImg, int neighbirSize)
{
	Mat DilatedImg = Mat::zeros(EdgeImg.size(), CV_8UC1);
	for (int i = neighbirSize; i < EdgeImg.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < EdgeImg.cols - neighbirSize; j++)
		{
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					if (EdgeImg.at<uchar>(i, j) == 0)
					{
						if (EdgeImg.at<uchar>(i + ii, j + jj) == 255)
						{
							DilatedImg.at<uchar>(i, j) = 255;
							break;
						}
					}
					else
						DilatedImg.at<uchar>(i, j) = 255;

				}
			}
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;

		}
	}

	return DilatedImg;

}

Mat Erosion(Mat EdgeImg, int neighbirSize)
{
	Mat ErodedImg = Mat::zeros(EdgeImg.size(), CV_8UC1);
	for (int i = neighbirSize; i < EdgeImg.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < EdgeImg.cols - neighbirSize; j++)
		{
			ErodedImg.at<uchar>(i, j) = 255;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					if (EdgeImg.at<uchar>(i, j) == 255)
					{
						if (EdgeImg.at<uchar>(i + ii, j + jj) == 0)
						{
							ErodedImg.at<uchar>(i, j) = 0;
							break;
						}
					}
					else if (EdgeImg.at<uchar>(i, j) == 0)
						ErodedImg.at<uchar>(i, j) = 0;



				}
			}
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;

		}
	}

	return ErodedImg;

}

Mat EqHist(Mat Grey)
{
	Mat EQImg = Mat::zeros(Grey.size(), CV_8UC1);
	// count
	int count[256] = { 0 };
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			count[Grey.at<uchar>(i, j)]++;


	// prob
	float prob[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);

	// accprob
	float accprob[256] = { 0.0 };
	accprob[0] = prob[0];
	for (int i = 1; i < 256; i++)
		accprob[i] = prob[i] + accprob[i - 1];
	// new = 255 * accprob 
	int newvalue[256] = { 0 };
	for (int i = 0; i < 256; i++)
		newvalue[i] = 255 * accprob[i];

	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			EQImg.at<uchar>(i, j) = newvalue[Grey.at<uchar>(i, j)];

	return EQImg;
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
int OTSU(Mat Grey)
{
	int count[256] = { 0 };
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			count[Grey.at<uchar>(i, j)]++;


	// prob
	float prob[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);

	// accprob
	float theta[256] = { 0.0 };
	theta[0] = prob[0];
	for (int i = 1; i < 256; i++)
		theta[i] = prob[i] + theta[i - 1];

	float meu[256] = { 0.0 };
	for (int i = 1; i < 256; i++)
		meu[i] = i * prob[i] + meu[i - 1];

	float sigma[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		sigma[i] = pow(meu[255] * theta[i] - meu[i], 2) / (theta[i] * (1 - theta[i]));

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
	for (int i = 1; i < 20; i++)
	{
		Mat img;
		std::string Path = "Resources/1_";
		std::string image = to_string(i);
		Path += image + ".jpg";

		img = imread(Path);
		Mat grey = RGB2Grey(img);
		Mat binary = Grey2Binary(grey, 128);
		Mat inverted = Inversion(grey);
		Mat stepped = Step(grey, 80, 140);

		Mat maxed = Max(grey, 2);
		Mat minned = Max(grey, 2);

		Mat histogrammed = EqHist(grey);
		Mat blurred = Avg(histogrammed, 1);

		float threshold = Average(blurred);

		std::cout << threshold << std::endl;

		float multiplier = (3.0f * threshold) / 255.0f;

		int edgeThreshold = 50;

		/*namedWindow("Trackbars", (640, 200));

		createTrackbar("Threshold", "Trackbars", &edgeThreshold, 100);*/

		//Mat edged = Edge(blurred, (int)(multiplier * edgeThreshold));
		Mat edged = Edge(blurred, 50);
		std::cout << (int)(multiplier * edgeThreshold) << std::endl;
		Mat eroded = Erosion(edged, 1);

		Mat dilated = Dilation(edged, 3);

		//Mat dilated = GetPlate(img);



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
		int minWidth = 20;
		int maxWidth = 200;
		int maxHeight = 500;
		int maxRatio = 1.5f;
		int stepValue = 5;
		int remainingSegments = contours.size();

		//while (remainingSegments > 1 || remainingSegments == 0)
		//{
		for (int i = 0; i < contours.size(); i++)
		{
			rect = boundingRect(contours[i]);

			float ratio = ((float)rect.width / (float)rect.height);
			/*for (int x = 0; x < rect.width; x++)
			{
				for (int y = 0; y < rect.height; y++)
				{
					if (rect.x - x >= dilated.cols - 1 || rect.y - y >= dilated.rows - 1)
					{
						continue;
					}
					if (rect.x - x < 1 || rect.y - y < 1)
					{
						continue;
					}

					dilated.at<uchar>(rect.x - x, rect.y - y) = 125;
				}
			}*/
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
				remainingSegments--;
			}
			else
			{
				plate = grey(rect);
			}
			//}
			minWidth += stepValue;
			maxWidth -= stepValue;
			//maxRatio -= 0.1f;
		}


		// RGB -> Grey -> Blur -> Edge -> (Erosion) -> Dilation

		//imshow("RGB image" + Path, img);
		//imshow("Greyscale image", grey);
		//imshow("Binary image", binary);
		//imshow("Inverted image", inverted);
		//imshow("Stepped image", stepped);
		//imshow("Blurred image", blurred);
		//imshow("Maxed image", maxed);
		//imshow("Minned image", minned);
		//imshow("Edged and blurred image", edged);
		//imshow("Dilated image", dilated);


		if (plate.rows != 0 && plate.cols != 0)
		{
			//imshow("Plate", plate);
		}

		int OTSUThreshold = OTSU(plate);

		Mat binarizedPlate = Grey2Binary(plate, OTSUThreshold);

		Mat binaryPlateCopy = binarizedPlate.clone();
		vector <vector<Point>> contours1;
		vector<Vec4i> hierarchy1;

		//binarizedPlate = Dilation(binarizedPlate, 1);
		//binarizedPlate = Erosion(binarizedPlate, 1);


		if (binarizedPlate.rows != 0 && binarizedPlate.cols != 0)
		{
			imshow(to_string(i), binarizedPlate);
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

		//imshow("Segmented plate" + Path, dst1);


		Mat character;
		vector<Mat> letters;
		for (int i = 0; i < contours1.size(); i++)
		{
			rect = boundingRect(contours1[i]);

			if (rect.height < 5)
			{
				//drawContours(dilated, contours, i, black, -1, 8, hierarchy);
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

		//imshow(to_string(i), dst);

		tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

		api->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK);
		api->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

		if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng"))
		{
			std::cout << "Could not initialize Tesseract!" << std::endl;
			exit(1);
		}


		if (!letters.empty())
		{
			for (int i = 0; i < letters.size(); i++)
			{
				resize(letters[i], letters[i], Size(letters[i].size().width * 4, letters[i].size().height * 4), 0, 0, INTER_LINEAR);

				letters[i] = Avg(letters[i], 3);

				letters[i] = Step(letters[i], 240, 255);

				api->SetImage(letters[i].data, letters[i].size().width, letters[i].size().height, letters[i].channels(), letters[i].step1());
				const char* letter = api->GetUTF8Text();
				std::cout << letter;
			}
		}

	}
	waitKey();



}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
