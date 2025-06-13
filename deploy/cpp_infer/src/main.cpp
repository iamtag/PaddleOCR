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

#include <opencv2/imgcodecs.hpp>

#include <include/args.h>
#include <include/paddlestructure.h>

#include <iostream>
#include <vector>
#include <include/json/json.h>

using namespace PaddleOCR;

#include <fstream>
#include <iomanip> // for std::setprecision

#include <windows.h>
#include <cstring>
#include "d:/gpu/cuda/include/cuda_runtime.h"

void save_result_json(const std::vector<std::vector<OCRPredictResult>>& ocr_results, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    ofs << "{\n";
    ofs << "   \"code\" : \"0\",\n";
    ofs << "   \"result\" : [\n";
    bool first = true;
    for (const auto& img_results : ocr_results) {
        // 对每个图片的结果进行排序
		std::vector<OCRPredictResult> ocr_result = img_results;
        Utility::sort_boxes(ocr_result);

        for (const auto& res : ocr_result) {
			// 如果分数小于等于0.7或者识别内容为空，则跳过
			if (res.score <= 0.7 || res.text.empty()) {
				continue;
			}
            // 去掉识别内容中的"
			std::string text = res.text;
			text.erase(std::remove(text.begin(), text.end(), '\"'), text.end());

            if (!first) ofs << ",\n";
            first = false;
            ofs << "      {\n";
            // 假设 box 是4个点，顺时针排列
            if (res.box.size() == 4) {
                ofs << "         \"P1\" : \"" << res.box[0][0] << "," << res.box[0][1] << "\",\n";
                ofs << "         \"P2\" : \"" << res.box[1][0] << "," << res.box[1][1] << "\",\n";
                ofs << "         \"P3\" : \"" << res.box[2][0] << "," << res.box[2][1] << "\",\n";
                ofs << "         \"P4\" : \"" << res.box[3][0] << "," << res.box[3][1] << "\",\n";
            }
            else {
                ofs << "         \"P1\" : \"\",\n";
                ofs << "         \"P2\" : \"\",\n";
                ofs << "         \"P3\" : \"\",\n";
                ofs << "         \"P4\" : \"\",\n";
            }
            ofs << "         \"score\" : " << std::setprecision(10) << res.score << ",\n";
            ofs << "         \"text\" : \"" << text << "\"\n";
            ofs << "      }";
        }
    }
    ofs << "\n   ]\n";
    ofs << "}\n";
    ofs.close();
}


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
    std::cout << "precision should be 'fp32'(default), 'fp16' or 'int8'. "
              << std::endl;
    exit(1);
  }
}

void ocr(std::vector<cv::String> &cv_all_img_names,
         std::vector<cv::String>& cv_all_dst_names) {
  PPOCR ocr;

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
    img_list.emplace_back(std::move(img));
    img_names.emplace_back(cv_all_img_names[i]);
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
  //save_result_json(ocr_results, "result.json");
}

void structure(std::vector<cv::String> &cv_all_img_names) {
  PaddleOCR::PaddleStructure engine;

  if (FLAGS_benchmark) {
    engine.reset_timer();
  }

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }

    std::vector<StructurePredictResult> structure_results = engine.structure(
        img, FLAGS_layout, FLAGS_table, FLAGS_det && FLAGS_rec);

    for (size_t j = 0; j < structure_results.size(); ++j) {
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

// 检查是否设置了use_gpu参数，如果未设置则自动检测cudnn64_9.dll
void auto_set_use_gpu(int argc, char** argv) {
    bool user_set_use_gpu = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strstr(argv[i], "--use_gpu") != nullptr) {
            user_set_use_gpu = true;
            break;
        }
    }
	std::cout << "[INFO] use_gpu flag:" << user_set_use_gpu << std::endl;
    if (!user_set_use_gpu) {
        int device_count = 0;
        cudaError_t status = cudaGetDeviceCount(&device_count);
        std::cout << "[DBG] status=" << status << ", cnt=" << device_count << std::endl;
        if (status != cudaSuccess || device_count == 0) {
            FLAGS_use_gpu = false;
            std::cout << "[INFO] not found cuda deivce, set use_gpu=false" << std::endl;
        }
    }
}

void append_pplib_to_path() {
    char old_path[4096] = { 0 };
    DWORD len = GetEnvironmentVariableA("PATH", old_path, sizeof(old_path));
    std::string new_path = old_path;
    if (!new_path.empty() && new_path.back() != ';') {
        new_path += ";";
    }
    new_path += "pplib/cudnn12.6;pplib/cuda12.6";
    SetEnvironmentVariableA("PATH", new_path.c_str());
}

int main(int argc, char **argv) {
  append_pplib_to_path();
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto_set_use_gpu(argc, argv);
  check_params();

  if (!Utility::PathExists(FLAGS_image_dir)) {
    std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
              << std::endl;
    exit(1);
  }

  //读取json文件内的json数据
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

  //std::vector<cv::String> cv_all_img_names;
  //cv::glob(FLAGS_image_dir, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

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
