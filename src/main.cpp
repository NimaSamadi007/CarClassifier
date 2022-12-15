#include <iostream>
#include "VehicleClassification.hpp"

int main(){
    VehicleClassification vecls{"../config.json"};

    cv::Mat input_img = cv::imread("../sample_imgs/00001.jpg", cv::IMREAD_COLOR);
    std::cout << input_img.size() << std::endl;
    cv::imshow("result", input_img);
    cv::waitKey(0);

    return 0;
}
