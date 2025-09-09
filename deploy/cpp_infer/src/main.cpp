// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

#include <include/args.h>
#include <include/paddleocr.h>
#include <include/paddlestructure.h>

#include <include/json/json.h>
#include <fstream>
#include <algorithm> // 添加这行来包含 std::replace
#include <include/socket_utils.h>

using namespace PaddleOCR;

void check_params() {
  if (FLAGS_det) {
    if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[det]: ./ppocr "
                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_rec) {
    std::cout
        << "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320',"
           "if you are using recognition model with PP-OCRv2 or an older "
           "version, "
           "please set --rec_image_shape='3,32,320"
        << std::endl;
    if (FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[rec]: ./ppocr "
                   "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_cls && FLAGS_use_angle_cls) {
    if (FLAGS_cls_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[cls]: ./ppocr "
                << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_table) {
    if (FLAGS_table_model_dir.empty() || FLAGS_det_model_dir.empty() ||
        FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[table]: ./ppocr "
                << "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--table_model_dir=/PATH/TO/TABLE_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_layout) {
    if (FLAGS_layout_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[layout]: ./ppocr "
                << "--layout_model_dir=/PATH/TO/LAYOUT_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" &&
      FLAGS_precision != "int8") {
    std::cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. "
              << std::endl;
    exit(1);
  }
}

void ocr_client_handle_one_file(const std::string& img_path, const std::string& dst_json_path) {
    SocketClient client(8866);
    client.connect();

    std::string json_string;
    if (img_path == "EXIT") {
        json_string = "EXIT";
    }
    else {
        // 将路径中的反斜杠替换为正斜杠，确保构造的 JSON 字符串有效
        std::string processed_img_path = img_path;
        std::string processed_dst_path = dst_json_path;
        std::replace(processed_img_path.begin(), processed_img_path.end(), '\\', '/');
        std::replace(processed_dst_path.begin(), processed_dst_path.end(), '\\', '/');

        // 使用处理后的路径构造 JSON 字符串
        json_string = "{\"img_path\":\"" + processed_img_path + "\", \"dst_json_path\":\"" + processed_dst_path + "\"}";
    }
    Utility::log_with_timestamp("[INFO] sending image path: ") << img_path << " dst_json_path: " << dst_json_path << std::endl;
    client.send(json_string);
    json_string = client.receive();
    if (json_string.empty()) {
        std::cerr << "[ERROR] No response from server." << std::endl;
        return;
    }
    Utility::log_with_timestamp("[INFO] ") << json_string << std::endl;
}

void ocr_client() {
	std::string image_dir = FLAGS_image_dir;
    if (Utility::is_json_file(image_dir)) {
        std::vector<cv::String> cv_all_img_names;
        std::vector<cv::String> cv_all_dst_names;
        if (Utility::parse_input_json(image_dir, cv_all_img_names, cv_all_dst_names)) {
            for (size_t i = 0; i < cv_all_img_names.size(); ++i) {
                ocr_client_handle_one_file(cv_all_img_names[i], cv_all_dst_names[i]);
            }
        }
    }
    else {
        ocr_client_handle_one_file(image_dir, FLAGS_output_json_path);
    }
}

void ocr_service() {
  //通过ocr一个测试样例图片，预加载所需资源
  std::string test_file = "textline.png";
  std::string application_path = Utility::get_application_path();
  if (!application_path.empty()) {
	  std::string model_dir = application_path + "/pplib";
	  if (FLAGS_det_model_dir.empty()) {
		  FLAGS_det_model_dir = model_dir + "/ch_ppocr_det_infer";
	  }
	  if (FLAGS_rec_model_dir.empty()) {
		  FLAGS_rec_model_dir = model_dir + "/ch_ppocr_rec_infer";
	  }
	  if (FLAGS_cls_model_dir.empty()) {
		  FLAGS_cls_model_dir = model_dir + "/ch_ppocr_cls_infer";
	  }
	  if (FLAGS_rec_char_dict_path.empty()) {
          FLAGS_rec_char_dict_path = model_dir + "/ppocr_keys_v1.txt";
	  }
	  test_file = model_dir + "/" + test_file;
  }
  else {
	  std::cerr << "[ERROR] Failed to get application path." << std::endl;
	  return;
  }

  SocketServer server(8866);
  server.start();
  PPOCR ocr = PPOCR();
  std::string json_string;
  std::vector<OCRPredictResult> ocr_results;
  std::string img_path, dst_json_path;

  if (Utility::PathExists(test_file)) {
	  cv::Mat img = cv::imread(test_file, cv::IMREAD_COLOR);
	  if (!img.data) {
		  std::cerr << "[ERROR] test image read failed! image path: "
			  << test_file << std::endl;
		  return;
	  }
	  ocr_results = ocr.ocr(img);
	  Utility::log_with_timestamp("[INFO] pre-load OCR resources with test image: ") << test_file << std::endl;
  }
  else {
	  Utility::log_with_timestamp("[WARNING] test image not found, skipping pre-load.") << std::endl;
  }

  while (true) {
	  Utility::log_with_timestamp("[INFO] waiting for client connection...") << std::endl;
    json_string = server.receive();
    if (json_string.empty()) {
      continue;
    }
    if (json_string == "EXIT") {
      break;
    }

    Json::Value input_json_value;
    Json::Reader reader;
    reader.parse(json_string, input_json_value);
    img_path = input_json_value["img_path"].asString();
    dst_json_path = input_json_value["dst_json_path"].asString();
    Utility::log_with_timestamp("[INFO] received image path: ") << img_path << " dst_json_path: " << dst_json_path << std::endl;

    int output_length = 0;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (!img.data) {
      json_string = "{\"code\": \"1\", \"message\": \"image read failed!\"}";//按标准API返回格式返回
      std::cerr << "[ERROR] image read failed! image path: "
                << img_path << std::endl;
    } else {
      ocr_results = ocr.ocr(img);
      if (!dst_json_path.empty()) {
        output_length = Utility::save_result_json(ocr_results, dst_json_path);
		if (output_length <= 0) {
			json_string = "{\"code\": \"1\", \"message\": \"save result json failed!\"}";
			std::cerr << "[ERROR] save result json failed! dst path: "
				<< dst_json_path << std::endl;
		}
		else {
			json_string = "{\"code\": \"0\", \"message\": \"success\", \"output_length\": " + std::to_string(output_length) + "}";
		}
      }
      else {
        json_string = Utility::ocr_results_to_string(ocr_results);
        output_length = json_string.length();
      }
    }
    server.send(json_string);
    Utility::log_with_timestamp("[INFO] output_length:") << output_length << std::endl;
  }
}

void ocr(std::vector<cv::String> &cv_all_img_names, std::vector<cv::String> &cv_all_dst_names) {
  PPOCR ocr = PPOCR();

  if (FLAGS_benchmark) {
    ocr.reset_timer();
  }

  std::vector<cv::Mat> img_list;
  std::vector<cv::String> img_names;
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }
    img_list.push_back(img);
    img_names.push_back(cv_all_img_names[i]);
  }

  std::vector<std::vector<OCRPredictResult>> ocr_results =
      ocr.ocr(img_list, cv_all_dst_names, FLAGS_det, FLAGS_rec, FLAGS_cls);

  for (int i = 0; i < img_names.size(); ++i) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    Utility::print_result(ocr_results[i]);
    if (FLAGS_visualize && FLAGS_det) {
      std::string file_name = Utility::basename(img_names[i]);
      cv::Mat srcimg = img_list[i];
      Utility::VisualizeBboxes(srcimg, ocr_results[i],
                               FLAGS_output + "/" + file_name);
    }
  }
  if (FLAGS_benchmark) {
    ocr.benchmark_log(cv_all_img_names.size());
  }
}

void structure(std::vector<cv::String> &cv_all_img_names) {
  PaddleOCR::PaddleStructure engine = PaddleOCR::PaddleStructure();

  if (FLAGS_benchmark) {
    engine.reset_timer();
  }

  for (int i = 0; i < cv_all_img_names.size(); i++) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }

    std::vector<StructurePredictResult> structure_results = engine.structure(
        img, FLAGS_layout, FLAGS_table, FLAGS_det && FLAGS_rec);

    for (int j = 0; j < structure_results.size(); j++) {
      std::cout << j << "\ttype: " << structure_results[j].type
                << ", region: [";
      std::cout << structure_results[j].box[0] << ","
                << structure_results[j].box[1] << ","
                << structure_results[j].box[2] << ","
                << structure_results[j].box[3] << "], score: ";
      std::cout << structure_results[j].confidence << ", res: ";

      if (structure_results[j].type == "table") {
        std::cout << structure_results[j].html << std::endl;
        if (structure_results[j].cell_box.size() > 0 && FLAGS_visualize) {
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          Utility::VisualizeBboxes(img, structure_results[j],
                                   FLAGS_output + "/" + std::to_string(j) +
                                       "_" + file_name);
        }
      } else {
        std::cout << "count of ocr result is : "
                  << structure_results[j].text_res.size() << std::endl;
        if (structure_results[j].text_res.size() > 0) {
          std::cout << "********** print ocr result "
                    << "**********" << std::endl;
          Utility::print_result(structure_results[j].text_res);
          std::cout << "********** end print ocr result "
                    << "**********" << std::endl;
        }
      }
    }
  }
  if (FLAGS_benchmark) {
    engine.benchmark_log(cv_all_img_names.size());
  }
}

int main(int argc, char **argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_start_server) {
      ocr_service();
      return 0;
  }
  if (FLAGS_ocr_client) {
	  ocr_client();
	  return 0;
  }
  check_params();

  if (!Utility::PathExists(FLAGS_image_dir)) {
    std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
              << std::endl;
    exit(1);
  }

  std::vector<cv::String> cv_all_img_names;
  std::vector<cv::String> cv_all_dst_names;
  Utility::parse_input_json(FLAGS_image_dir, cv_all_img_names, cv_all_dst_names);  

  if (!Utility::PathExists(FLAGS_output)) {
    Utility::CreateDir(FLAGS_output);
  }
  if (FLAGS_type == "ocr") {
    ocr(cv_all_img_names, cv_all_dst_names);
  } else if (FLAGS_type == "structure") {
    structure(cv_all_img_names);
  } else {
    std::cout << "only value in ['ocr','structure'] is supported" << std::endl;
  }
}
