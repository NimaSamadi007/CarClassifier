#include <iostream>
#include <exception>
#include <stdlib.h>
#include "VehicleClassification.hpp"

void runOneShotMode(char* img_path);
void runContinousMode();
void handleArgs(int argc, char* argv[]);
std::string runModel(VehicleClassification& model, cv::Mat input_img);

int main(int argc, char* argv[]){
    try {
        if(argc >= 2)
            handleArgs(argc, argv);
        else
            throw std::invalid_argument("Working mode is not provided");
    } catch(const std::exception& e) {
        std::cout << "Exception caught!\n";
        std::cout << e.what() << "\n";
        std::cout << "Program terminated" << std::endl;
    }

    return 0;
}

void handleArgs(int argc, char* argv[]){
    char* p;
    long mode = strtol(argv[1], &p, 10);
    if(*p == '\0'){
        if(mode == 0){
            if(argc == 3)
                runOneShotMode(argv[2]);
            else
                throw std::invalid_argument("Image path must be provided in one-shot mode");
        } else {
            runContinousMode();
        }
    } else {
        throw std::invalid_argument("Invalid character found for second argument");
    }
}

void runOneShotMode(char* img_path){
    VehicleClassification vecls{"../config.json"};
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    auto label = runModel(vecls, img);
    cv::putText(img, label,
                cv::Point(20, 30), cv::FONT_ITALIC,
                1, cv::Scalar(0, 255, 0), 3);
    cv::imshow("result", img);
    cv::waitKey(0);
}

void runContinousMode(){
    VehicleClassification vecls{"../config.json"};
    cv::VideoCapture cap(0);
    if(cap.isOpened()){
        bool end_loop = false;
        cv::Mat img;
        std::string label;
        while(!end_loop){
            cap >> img;
            float model_output = 0;
            if(!img.empty()){
                auto label = runModel(vecls, img);
                cv::putText(img, label,
                            cv::Point(20, 30), cv::FONT_ITALIC,
                            1, cv::Scalar(0, 255, 0), 3);
                cv::imshow("result", img);
                int rec_key = cv::waitKey(25);
                if (rec_key == int('q'))
                    end_loop = true;
            }
        }
    } else {
        throw std::runtime_error("Unable to open camera");
    }
}

std::string runModel(VehicleClassification& model, cv::Mat input_img){
    auto model_output = model.inference(input_img);
    std::string label;
    if(model_output >= 0)
        label = "car";
    else
        label = "nocar";
    std::cout << "Found " << label << " in image\n";
    return label;
}
