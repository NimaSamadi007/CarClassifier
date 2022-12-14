#ifndef _VEHICLE_CLASSIFICATION_HPP_
#define _VEHICLE_CLASSIFICTAION_HPP_

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include "json.hpp"

namespace nl=nlohmann;

class VehicleClassification{
private:
    // Configuratios
    std::string model_path;
    unsigned int num_cpu_threads;
    bool use_gpu;

    // Variables
    std::shared_ptr<Ort::Env> env_cls;
    std::shared_ptr<Ort::Session> session_cls;

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

public:
    VehicleClassification(const std::string config_file_path);

};

#endif //_VEHICLE_DETECTION_HPP_
