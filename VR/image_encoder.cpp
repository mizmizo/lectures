#include <iostream>
#include <vector>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <objdetect/objdetect.hpp>

int main (int argc, char **argv)
{
  if(argc != 3){
    std::cerr << "enter image file name and cascade file name!" << std::endl;
    return 0;
  }
  int i;
  cv::Mat src_img, src_gray;
  std::string image_name = std::string(argv[1]);
  std::string cascade_name = std::string(argv[2]);

  src_img = cv::imread(image_name,1);
  if(src_img.empty()) {
    std::cerr << "cannot load image" << std::endl;
    return 0;
  }

  cv::CascadeClassifier cascade;
  cascade.load(cascade_name);
  if(cascade.empty()) {
    std::cerr << "cannot load cascade" << std::endl;
    return 0;
  }

  cv::cvtColor (src_img, src_gray, CV_RGB2GRAY);
  cv::equalizeHist (src_gray, src_gray);

  std::vector<cv::Rect> objects;
  cascade.detectMultiScale(src_gray, objects, 1.1, 3);
  std::vector<cv::Rect>::const_iterator iter = objects.begin();
  std::cout << "find objects : " << objects.size() << std::endl;
  while(iter!=objects.end()) {
    std::cout << "(x, y, width, height) = (" << iter->x << ", " << iter->y << ", " << iter->width << ", " << iter->height << ")" << std::endl;
    cv::rectangle(src_img, cv::Rect(iter->x, iter->y, iter->width, iter->height), cv::Scalar(0, 0, 255), 2);
    ++iter;
  }

  cv::imwrite("detect_result.jpg", src_img);
  cv::namedWindow("Objects Detection", CV_WINDOW_AUTOSIZE);
  cv::imshow("Objects Detection", src_img);
  cv::waitKey (0);

  cv::destroyWindow ("Objects Detection");
  return 0;
}
