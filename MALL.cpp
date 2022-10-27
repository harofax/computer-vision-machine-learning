#include <iostream>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

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


		}
	}
	return ErodedImg;
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

int main()
{
	Mat img;
	std::string Path = "Resources/1.jpg";
	img = imread(Path);
	Mat GreyImg = RGB2Grey(img);
	imshow("Grey image", GreyImg);

	Mat EdgeImg = Edge(GreyImg, 50);
	imshow("Edge image", EdgeImg);


	Mat binaryImg = Grey2Binary(GreyImg, 128);
	imshow("Binary image", binaryImg);

	Mat AvgImg = Avg(GreyImg, 1);
	imshow("3*3 Blur image", AvgImg);

	Mat EdgeImg2 = Edge(AvgImg, 50);
	imshow("Edge with Blur image", EdgeImg2);

	Mat ErodedImg = ErosionOpt(EdgeImg2, 1);
	imshow("Eroded 3*3", ErodedImg);

	Mat DilatedImg3 = Dilation(ErodedImg, 15);
	imshow("Dilation after erosion 9*9 ", DilatedImg3);

	Mat DilatedImg = Dilation(EdgeImg2, 3);
	imshow("Dilation 3*3 ", DilatedImg);

	Mat DilatedImg2 = Dilation(EdgeImg2, 4);
	imshow("Dilation 7*7 ", DilatedImg2);
	Mat DilatedImgCpy;
	DilatedImgCpy = DilatedImg2.clone();
	vector<vector<Point>> contours1;
	vector<Vec4i> hierachy1;
	findContours(DilatedImg2, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	Mat dst = Mat::zeros(GreyImg.size(), CV_8UC3);

	if (!contours1.empty())
	{
		for (int i = 0; i < contours1.size(); i++)
		{
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
		}
	}

	Mat plate;
	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);
	for (int i = 0; i < contours1.size(); i++)
	{
		rect = boundingRect(contours1[i]);
		float ratio = (float)rect.width / rect.height;
		if (rect.width < 40
			|| rect.y < DilatedImgCpy.rows * 0.1f
			|| rect.y > DilatedImgCpy.rows * 0.9f
			|| rect.height > 100
			|| ratio < 1.5f)
		{
			drawContours(DilatedImgCpy, contours1, i, black, -1, 8, hierachy1);
		}
		else
		{
			plate = GreyImg(rect);
		}
	}

	imshow("Filtered Image", DilatedImgCpy);
	if (plate.rows != 0 && plate.cols != 0)
	{
		imshow("Plate", plate);
	}

	Mat binaryImg2 = Grey2Binary(AvgImg, 128);


	waitKey();

	std::cout << "Hello World!\n";
}



Mat Dilation(Mat imgBin, int range = 1)
{
	Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

	for (int i = range; i < imgBin.rows - range; i++)
	{
		for (int j = range; j < imgBin.cols - range; j++)
		{
			// skip if pixel is already white
			if (imgBin.at<uchar>(i, j) == 255)
			{
				output.at<uchar>(i, j) = 255;
				continue;
			}

			for (int ii = -range; ii <= range; ii++)
			{
				for (int jj = -range; jj <= range; jj++)
				{
					int value = imgBin.at<uchar>(i + ii, j + jj);

					// make pixel white if it has white neighbours
					if (value == 255)
					{
						output.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}
	return output;
}

cv::Mat dilate(cv::Mat& img, int border)
{
	auto hit = [img, border](int startRow, int startCol)
	{
		for (int row = startRow - border; row <= startRow + border; row++)
		{
			for (int col = startCol - border; col <= startCol + border; col++)
			{
				if (img.at<uchar>(row, col) == 255)
					return true;
			}
		}
		return false;
	};

	cv::Mat dilatedImg = cv::Mat::zeros(img.size(), CV_8UC1);
	for (int row = border; row < img.rows - border; row++)
	{
		for (int col = border; col < img.cols - border; col++)
		{
			int pixel = img.at<uchar>(row, col);
			if (pixel == 255 || pixel == 0 && hit(row, col))
				dilatedImg.at<uchar>(row, col) = 255;
		}
	}
	return dilatedImg;
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
