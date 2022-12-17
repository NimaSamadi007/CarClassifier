#ifndef _VEHICLE_CLASSIFICATION_HPP_
#define _VEHICLE_CLASSIFICTAION_HPP_

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <vector>
#include "json.hpp"

namespace nl=nlohmann;

class VehicleClassification{
private:
    // Configuratios
    std::string model_path;
    unsigned int num_cpu_threads;
    bool use_gpu;
    static constexpr int64_t batch_size = 1;
    static constexpr int64_t input_width = 160;
    static constexpr int64_t input_height = 160;
    static constexpr int64_t input_tensor_size = batch_size*input_width*input_height*3;
    std::vector<int64_t> input_shapes{batch_size, input_height, input_width, 3};

    // Variables
    std::shared_ptr<Ort::Env> env_cls;
    std::shared_ptr<Ort::Session> session_cls;
    float* image_blob;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    // Methods
    void readConfigFile(const std::string config_file_path);
    void configureParams(nl::json config_json);
    void setEnvAndSession();
    Ort::SessionOptions setCPUSessionOptions();
    OrtCUDAProviderOptions setGPUSessionOptions();
    void setInputNamesAndShape();
    void setOutputNamesAndShape();
    void preProc(cv::Mat img);
    float runModel(void);
public:
    VehicleClassification(const std::string config_file_path);
    ~VehicleClassification();
    float inference(cv::Mat img);
};

#endif //_VEHICLE_DETECTION_HPP_
