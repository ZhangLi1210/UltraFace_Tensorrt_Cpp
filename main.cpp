#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <iostream>
#include<vector>
#include <fstream>
#include <cassert>
#include <string>
#include "include/UltraFace.h"



int main() {
    std::string model = "../engine/version-RFB-320.engine";

    auto UltraFace_dector = new UltraFace(model, 320, 240, 0.6, 0.1);

    cv::VideoCapture cap;
    cv::Mat frame;
    //cap.open("track_text_vidio.mp4");
    cap.open(0);
    int count = 0;
    std::vector<FaceInfo> face_info;
    for (;;) {
        count +=1;
        // Read a new frame.
        cap >> frame;
        if (frame.empty())
            break;
        //检测速度较快,其实并不需要每一帧都进行检测    
        if(count == 3){
            face_info.clear();
            auto start = std::chrono::high_resolution_clock::now();
            UltraFace_dector->detect(frame, face_info);
            // 获取结束时间点
            auto end = std::chrono::high_resolution_clock::now();
            // 计算时间差
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            // 输出运行时间（毫秒）
            std::cout << "fps: " << 1000/duration.count() << std::endl;
            count =0;
        }
        for (int i = 0; i < face_info.size(); i++) {
            //std::cout<<"1"<<std::endl;
            auto face = face_info[i];
            //UltraFace的框会大一点,这里将宽度进行了一定程度的缩放.
            cv::Point pt1(face.x1+(face.x2-face.x1)/10, face.y1);
            cv::Point pt2(face.x2-(face.x2-face.x1)/9, face.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 5);
        }
        cv::imshow("UltraFace", frame);
        cv::waitKey(4);
    }
    cv::destroyWindow("UltraFace");
    cap.release();
    delete UltraFace_dector;
    return 0;
}

