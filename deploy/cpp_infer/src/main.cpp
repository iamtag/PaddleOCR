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

std::string ocr_results_to_string(const std::vector<OCRPredictResult> &ocr_results) {
  /*参考save_result_json函数代码*/
  Json::Value root;
  Json::Value results;
  for (const auto& res : ocr_results) {
    std::string res_text = res.text;
    res_text = Utility::replace_all(res_text, "\"", "");
    if (res.score <= 0.7 || res_text.empty()) {
      continue; // Skip results with low confidence or empty text
    }
    
    Json::Value result;
    result["text"] = Json::Value(res_text);
    result["score"] = Json::Value(res.score);
    // p1, p2, p3, p4 stand for
    // p1------------p2
    //  |             |
    //  |             |
    // p4------------p3
    if (res.box.size() == 4) {
      result["P1"] = Json::Value(std::to_string(res.box[0][0]) + "," + std::to_string(res.box[0][1]));
      result["P2"] = Json::Value(std::to_string(res.box[1][0]) + "," + std::to_string(res.box[1][1]));
      result["P3"] = Json::Value(std::to_string(res.box[2][0]) + "," + std::to_string(res.box[2][1]));
      result["P4"] = Json::Value(std::to_string(res.box[3][0]) + "," + std::to_string(res.box[3][1]));
    }
    else {
      result["P1"] = result["P2"] = result["P3"] = result["P4"] = Json::Value("");
    }
    results.append(result);
  }
  root["result"] = Json::Value(results);
  root["code"] = Json::Value("0");
  root["message"] = Json::Value("success");
  Json::StyledWriter sw;
  return sw.write(root);
}

void ocr_client() {
    SocketClient client(8866);
    client.connect();

    std::string img_path = FLAGS_image_dir;
    std::string dst_json_path = FLAGS_output_json_path;
    std::string json_string;
	if (img_path == "EXIT") {
		json_string = "EXIT";
    }
    else {
        json_string = "{\"img_path\":\"" + img_path + "\", \"dst_json_path\":\"" + dst_json_path + "\"}";
    }
	Utility::log_with_timestamp("[INFO] sending image path: ") << img_path << " dst_json_path: " << dst_json_path << std::endl;
	client.send(json_string);
}

void ocr_service() {
  SocketServer server(8866);
  server.start();
  PPOCR ocr = PPOCR();
  std::string json_string;
  std::vector<OCRPredictResult> ocr_results;
  std::string img_path, dst_json_path;
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

    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (!img.data) {
      json_string = "{\"code\": \"1\", \"message\": \"image read failed!\"}";//按标准API返回格式返回
      std::cerr << "[ERROR] image read failed! image path: "
                << img_path << std::endl;
    } else {
      ocr_results = ocr.ocr(img);
      if (!dst_json_path.empty()) {
		  Utility::save_result_json(ocr_results, dst_json_path);
          json_string = ""; //调用server.send("")关闭socket连接
	  }
	  else {
		  json_string = ocr_results_to_string(ocr_results);
	  }
    }
    server.send(json_string);
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
  if (FLAGS_ocr_server) {
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
  Json::Reader jsonreader;
  Json::Value root;
  std::ifstream in(FLAGS_image_dir, std::ios::binary);

  if (!in.is_open()) {
      std::cerr << "[ERROR] Error opening file! image_dir: " << FLAGS_image_dir << std::endl;
      exit(1);
  }
  if (jsonreader.parse(in, root))
  {
      for (unsigned int i = 0; i < root["files"].size(); i++)
      {
          std::string src = root["files"][i]["src"].asString();
          cv_all_img_names.push_back(cv::String(src.c_str()));
          std::string dst = root["files"][i]["dst"].asString();
          cv_all_dst_names.push_back(cv::String(dst.c_str()));
      }
  }
  in.close();

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
