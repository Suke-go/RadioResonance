cv::Mat gray;
cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

cv::Mat thresh;
cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 2);

std::vector<cv::Point2f> imagePoints, worldPoints;
cv::perspectiveTransform(imagePoints, worldPoints, homography);
