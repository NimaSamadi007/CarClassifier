#include "VehicleClassification.hpp"
#include <fstream>
#include <stdlib.h>
#include <assert.h>

VehicleClassification::VehicleClassification(const std::string config_file_path){
    readConfigFile(config_file_path);
    setEnvAndSession();
    setInputNamesAndShape();
    setOutputNamesAndShape();
    // Allocate image blob memory:
    this->image_blob = new float[input_tensor_size];
    std::cout << "VehicleClassification instance initialized successfully" << std::endl;
}

void VehicleClassification::readConfigFile(const std::string config_file_path){
    // Read config file
    nl::json config_json;
    std::ifstream config_file{config_file_path};
    config_file >> config_json;
    configureParams(config_json);
}

void VehicleClassification::configureParams(nl::json config_json){
    model_path = config_json["model_path"];
    num_cpu_threads = config_json["num_cpu_threads"];
    if(config_json["use_gpu"]){
        std::cout << "Using GPU for model inference" << std::endl;
        use_gpu = true;
    } else {
        std::cout << "Using CPU for model inference" << std::endl;
        use_gpu = false;
    }
}

void VehicleClassification::setEnvAndSession(){
    env_cls = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "VECls");
    Ort::SessionOptions sess_opts = setCPUSessionOptions();    
    if (use_gpu){
        OrtCUDAProviderOptions gpu_opts = setGPUSessionOptions();
        sess_opts.AppendExecutionProvider_CUDA(gpu_opts);
    }
    session_cls = std::make_shared<Ort::Session>(*env_cls, model_path.c_str(), sess_opts);
}

Ort::SessionOptions VehicleClassification::setCPUSessionOptions(){
    Ort::SessionOptions sess_opts;
    sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sess_opts.SetIntraOpNumThreads(num_cpu_threads);

    return sess_opts;
}

OrtCUDAProviderOptions VehicleClassification::setGPUSessionOptions(){
    OrtCUDAProviderOptions gpu_options;
	gpu_options.device_id = 0;
	gpu_options.arena_extend_strategy = 0;
	gpu_options.do_copy_in_default_stream = 1;
	gpu_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;

    return gpu_options;
}

void VehicleClassification::setInputNamesAndShape(){
    Ort::AllocatorWithDefaultOptions allocator;
    int num_inputs = session_cls->GetInputCount();

    for(size_t i = 0; i < num_inputs; i++){
        char* current_input_name = session_cls->GetInputName(i, allocator);
        input_names.push_back(current_input_name);
    }
}

void VehicleClassification::setOutputNamesAndShape(){
    Ort::AllocatorWithDefaultOptions allocator;
    int num_outputs = session_cls->GetOutputCount();

    for(size_t i = 0; i < num_outputs; i++){
        char* current_output_name = session_cls->GetOutputName(i, allocator);
        output_names.push_back(current_output_name);
    }
}

float VehicleClassification::inference(cv::Mat img){
    preProc(img);
    return runModel();
}

void VehicleClassification::preProc(cv::Mat img){
    // Scale input to 160x160
    cv::Mat resized_img, float_img;
    cv::resize(img, resized_img, cv::Size(input_width, input_height), cv::INTER_AREA);
    // Convert to float
    resized_img.convertTo(float_img, CV_32FC3);
    // Populate image_blob
    memcpy(image_blob, float_img.ptr<float>(0), sizeof(float)/sizeof(char) * input_tensor_size);
}

float VehicleClassification::runModel(){
    std::vector<float> input_tensor_values(image_blob, image_blob+input_tensor_size);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, 
                                                              input_tensor_values.data(),
                                                              input_tensor_size, 
                                                              input_shapes.data(),
                                                              input_shapes.size());
    assert(input_tensor.IsTensor());
    std::vector<Ort::Value> output_tensor = session_cls->Run(Ort::RunOptions{nullptr},
                                                             input_names.data(),
                                                             &input_tensor,
                                                             1,
                                                             output_names.data(),
                                                             1);
    float* raw_output = output_tensor[0].GetTensorMutableData<float>();
    return *raw_output;
}

VehicleClassification::~VehicleClassification(){
    delete[] this->image_blob;
}
