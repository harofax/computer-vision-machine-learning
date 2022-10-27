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

Mat Erosion(Mat edge, int neighbor_size)
{
	Mat erodedImage = Mat::zeros(edge.size(), CV_8UC1);

	for (int row = neighbor_size; row < edge.rows - neighbor_size; row++)
	{
		for (int col = neighbor_size; col < edge.cols - neighbor_size; col++)
		{
			//erodedImage.at<uchar>(row, col) = 255;
			erodedImage.at<uchar>(row, col) = edge.at<uchar>(row, col);

			for (int neighbor_row = -neighbor_size; neighbor_row < neighbor_size; neighbor_row++)
			{
				for (int neighbor_col = -neighbor_size; neighbor_col < neighbor_size; neighbor_col++)
				{
					if (edge.at<uchar>(row + neighbor_row, col + neighbor_col) == 0)
					{
						erodedImage.at<uchar>(row, col) = 0;
						break;
					}

				}
			}

		}
	}
	return erodedImage;
}

Mat Dilation(Mat edge, int neighbor_size)
{
	Mat dilatedImage = Mat::zeros(edge.size(), CV_8UC1);

	for (int row = neighbor_size; row < edge.rows - neighbor_size; row++)
	{
		for (int col = neighbor_size; col < edge.cols - neighbor_size; col++)
		{

			for (int neighbor_row = -neighbor_size; neighbor_row < neighbor_size; neighbor_row++)
			{
				for (int neighbor_col = -neighbor_size; neighbor_col < neighbor_size; neighbor_col++)
				{
					if (edge.at<uchar>(row + neighbor_row, col + neighbor_col) == 255)
					{
						dilatedImage.at<uchar>(row, col) = 255;
						break;
					}

				}
			}

		}
	}
	return dilatedImage;
}

Mat FindEdge(Mat grey, int threshold)
{
	Mat edgedImage = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 1; i < grey.rows - 1; i++)
	{
		for (int j = 1; j < grey.cols - 1; j++)
		{
			int averageLeft = (grey.at<uchar>(i - 1, j - 1) + grey.at<uchar>(i, j - 1) + grey.at<uchar>(i + 1, j - 1)) / 3;
			int averageRight = (grey.at<uchar>(i - 1, j + 1) + grey.at<uchar>(i, j + 1) + grey.at<uchar>(i + 1, j + 1)) / 3;

			if (abs(averageLeft - averageRight) > threshold)
			{
				edgedImage.at<uchar>(i, j) = 255;
			}

		}
	}
	return edgedImage;
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

Mat Step(Mat grey, int lowPass, int highPass)
{
	Mat steppedImage = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			if (grey.at<uchar>(i, j) >= lowPass && grey.at<uchar>(i, j) <= highPass)
			{
				steppedImage.at<uchar>(i, j) = 255;
			}
		}
	}
	return steppedImage;
}

Mat Inversion(Mat grey)
{
	Mat invertedImage = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			invertedImage.at<uchar>(i, j) = 255 - grey.at<uchar>(i, j);
		}
	}
	return invertedImage;
}

Mat GreyToBinary(Mat grey, int threshold)
{
	Mat binaryImage = Mat::zeros(grey.size(), CV_8UC1);

	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			if (grey.at<uchar>(i, j) > threshold)
			{
				binaryImage.at<uchar>(i, j) = 255;
			}
		}
	}
	return binaryImage;
}

Mat RGBToGrey(Mat RGB)
{
	Mat greyscaleImage = Mat::zeros(RGB.size(), CV_8UC1);

	for (int i = 0; i < RGB.rows; i++)
	{
		for (int j = 0; j < RGB.cols * 3; j += 3)
		{
			greyscaleImage.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
		}
	}

	return greyscaleImage;
}

Mat GetPlate(Mat img)
{
	Mat greyImg = RGBToGrey(img);
	//imshow("Grey image", greyImg);

	Mat equalizedImg = EqualizeHistogram(greyImg);
	//imshow("Equalized image", equalizedImg);

	Mat blurredImg = Blur(equalizedImg, 1);
	//imshow("Blurred/Averaged image", blurredImg);

	Mat edgedImg = FindEdge(blurredImg, 50);
	//imshow("Edged image", blurredImg);

	Mat erodedImg = Erosion(edgedImg, 1);
	//imshow("Eroded image", erodedImg);

	Mat dilatedImg = Dilation(erodedImg, 8);
	//imshow("Dilated image", dilatedImg);

	return dilatedImg;
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

int main()
{

	Mat img;
	img = imread("C:\\Users\\Victor\\Desktop\\cars\\1_2.jpg");
	Mat grey = RGBToGrey(img);
	Mat binary = GreyToBinary(grey, 128);
	Mat inverted = Inversion(grey);
	Mat stepped = Step(grey, 80, 140);

	Mat maxed = Max(grey, 2);
	Mat minned = Max(grey, 2);

	Mat histogrammed = EqualizeHistogram(grey);
	Mat blurred = Blur(histogrammed, 1);

	float threshold = Average(blurred);

	std::cout << threshold << std::endl;

	float multiplier = (0.5f * threshold) / 255.0f;

	int edgeThreshold = 50;

	/*namedWindow("Trackbars", (640, 200));

	createTrackbar("Threshold", "Trackbars", &edgeThreshold, 100);*/

	Mat edged = FindEdge(blurred, (int)(multiplier * edgeThreshold));
	//Mat edged = FindEdge(blurred, 50);
	std::cout << (int)(multiplier * edgeThreshold) << std::endl;
	Mat eroded = Erosion(edged, 1);

	Mat dilated = Dilation(edged, 4);

	//Mat dilated = GetPlate(img);

	//int minWidth = 100;
	//int maxWidth = 300;
	//int maxHeight = 500;
	//int remainingSegments = 0;
	//remainingSegments = contours.size();

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


	for (int i = 0; i < contours.size(); i++)
	{
		rect = boundingRect(contours[i]);

		float ratio = ((float)rect.width / (float)rect.height);

		if (rect.width < 40
			|| rect.width > 100
			|| rect.height > 100
			|| rect.x <= (grey.rows * 0.1f)
			|| rect.x >= (grey.rows * 0.9f)
			|| rect.y >= (grey.cols * 0.9f)
			|| rect.y <= (grey.cols * 0.1f)
			|| ratio < 1.5f)
		{
			drawContours(dilated, contours, i, black, -1, 8, hierarchy);
		}
		else
		{
			plate = grey(rect);
		}
	}


	// RGB -> Grey -> Blur -> Edge -> (Erosion) -> Dilation

	imshow("RGB image", img);
	imshow("Greyscale image", grey);
	//imshow("Binary image", binary);
	//imshow("Inverted image", inverted);
	//imshow("Stepped image", stepped);
	//imshow("Blurred image", blurred);
	//imshow("Maxed image", maxed);
	//imshow("Minned image", minned);
	imshow("Edged and blurred image", edged);
	imshow("Dilated image", dilated);


	if (plate.rows != 0 && plate.cols != 0)
	{
		imshow("Plate", plate);
	}

	int OTSUThreshold = OTSU(plate);

	Mat binarizedPlate = GreyToBinary(plate, OTSUThreshold);


	Mat binaryPlateCopy = binarizedPlate.clone();
	vector <vector<Point>> contours1;
	vector<Vec4i> hierarchy1;

	//binarizedPlate = Dilation(binarizedPlate, 1);
	//binarizedPlate = Erosion(binarizedPlate, 1);


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

	//imshow("Segmented image", dst);

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

			letters[i] = Blur(letters[i], 3);

			letters[i] = Step(letters[i], 240, 255);

			api->SetImage(letters[i].data, letters[i].size().width, letters[i].size().height, letters[i].channels(), letters[i].step1());
			const char* letter = api->GetUTF8Text();
			std::cout << letter;
		}
	}



	waitKey();
}