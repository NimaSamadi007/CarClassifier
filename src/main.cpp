#include <iostream>
#include <exception>
#include "VehicleClassification.hpp"

int main(int argc, char* argv[]){
    try {
        if(argc != 2)
            throw std::invalid_argument("Enter input image path");
        VehicleClassification vecls{"../config.json"};
        cv::Mat input_img = cv::imread(argv[1], cv::IMREAD_COLOR);
        auto model_output = vecls.inference(input_img);
        std::string label;
        if(model_output >= 0)
            label = "car";
        else
            label = "nocar";
        cv::putText(input_img, label,
                    cv::Point(20, 30), cv::FONT_ITALIC,
                    1, cv::Scalar(0, 255, 0), 3);
        cv::imshow("result", input_img);
        cv::waitKey(0);
    } catch(const std::exception& e) {
        std::cout << "Exception caught!\n";
        std::cout << e.what() << "\n";
        std::cout << "Program terminated" << std::endl;
    }

    return 0;
}
