#include <opencv2/opencv.hpp>
#include <stdio.h>

#define OUT_VIDEO_FILE "sample_video_output.avi"

std::tuple<cv::Mat, double> cuttingPicture(cv::Mat input, cv::CascadeClassifier cascade, cv::CascadeClassifier cascade_eye);

int main(int argc, char *argv[])
{
  // 1. load classifier
  std::string cascadeName = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"; //Haar-like
  std::string cascadeName_eye = "/usr/share/opencv/haarcascades/haarcascade_eye.xml"; //目の認識
  cv::CascadeClassifier cascade, cascade_eye;
  if(!cascade.load(cascadeName)){
    printf("ERROR: cascadeFile not found\n");
    return -1;
  }
  if(!cascade_eye.load(cascadeName_eye)){
    printf("ERROR: cascade_eyeFile not found\n");
    return -1;
  }

  // 2. initialize VideoCapture
  cv::Mat frame;
  cv::Mat before;
  cv::Mat input_pic;
  cv::VideoCapture cap;
  cap.open(0);
  cap >> frame;
  cap >> before;
  double pic_mean;

  //鼻と口の画像
  cv::Mat input;
  

  if (argc == 1){  
    const char *preset_file = "nose_and_mouth.JPG";
    input_pic = cv::imread(preset_file, 1);
    if(input_pic.empty()){
        fprintf(stderr, "cannot open %s\n", preset_file);
        exit(0);
    }
    pic_mean = cv::mean(input_pic)[0];
  }
  else{
    const char *preset_file = argv[1];
    input = cv::imread(preset_file, 1);
    if(input.empty()){
        fprintf(stderr, "cannot open %s\n", preset_file);
        exit(0);
    }
    //画像の切り取り
    std::tie(input_pic, pic_mean) = cuttingPicture(input, cascade, cascade_eye); //画像の切り取りと明るさを取得
    cv::imshow("input", input_pic);
  }
  
  // 3. prepare window and trackbar
  //cv::namedWindow("after", 1);
  //cv::namedWindow("before", 1);

  double scale = 4.0;
  cv::Mat gray, smallImg(cv::saturate_cast<int>(frame.rows/scale), 
                cv::saturate_cast<int>(frame.cols/scale), CV_8UC1);

  for(;;){

    // 4. capture frame
    cap >> frame;
    cap >> before;

    //convert to gray scale
    cv::cvtColor(frame, gray, CV_BGR2GRAY);

    // 5. scale-down the image
    cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);

    // 6. detect face using Haar-classifier
    std::vector<cv::Rect> faces;
    std::vector<cv::Rect> eyes;
    // multi-scale face searching
    // image, size, scale, num, flag, smallest rect
    cascade.detectMultiScale(smallImg, faces,
                    1.01,  //大きくするとモザイク範囲ちいさくなる(2.1)
                    2, //大きくするとモザイク判定がゆっくりになる(40ぐらいから)
                    CV_HAAR_SCALE_IMAGE,
                    cv::Size(30, 30));

    // 7. mosaic(pixelate) face-region
    for(int i = 0; i < faces.size(); i++){
      cv::Point center;
      int radius;
      center.x = cv::saturate_cast<int>((faces[i].x + faces[i].width * 0.5) * scale);
      center.y = cv::saturate_cast<int>((faces[i].y + faces[i].height * 0.5) * scale);
      radius = cv::saturate_cast<int>((faces[i].width + faces[i].height) * 0.25 * scale);
      //mosaic
      cv::Rect face_rect(center.x - radius, center.y - radius, radius * 2, radius);
      double face_width = face_rect.width;
      cv::Mat faceRegion = frame(face_rect);
      cv::rectangle(frame, face_rect, cv::Scalar(0, 0, 255), 2);

      cascade_eye.detectMultiScale(faceRegion, eyes,
                    1.1,  //大きくするとモザイク範囲ちいさくなる(2.1)
                    2, //大きくするとモザイク判定がゆっくりになる(40ぐらいから)
                    CV_HAAR_SCALE_IMAGE,
                    cv::Size(30, 30));
      double min_x = 1000000.0;
      double min_y = 1000000.0;
      double min_width = 1000000.0;
      double min_height = 1000000.0;
      for (auto eyeRect:eyes){
          cv::Rect eye_rect(face_rect.x + eyeRect.x, face_rect.y + eyeRect.y, eyeRect.width, eyeRect.height);
          cv::rectangle(frame, eye_rect, cv::Scalar(0, 255, 0), 2);
          if (min_x > eyeRect.x){
              min_x = eyeRect.x;
              min_y = eyeRect.y;
              min_width = eyeRect.width;
              min_height = eyeRect.height;
          }

          cv::resize(input_pic, input_pic, cv::Size(), face_width/input_pic.cols, face_width/input_pic.cols, cv::INTER_LINEAR);
          cv::Mat roi = frame(cv::Rect(face_rect.x, face_rect.y + min_y + min_height, input_pic.cols, input_pic.rows));

          //明るさを測定 //目の間で測定したい
          cv::Rect center_rect(face_rect.x + min_x + min_width, face_rect.y + min_y, 10, min_height);
          cv::Mat center = frame(center_rect).clone();
          double video_mean = cv::mean(center)[0];
          cv::Mat pic_after;
          input_pic.convertTo(pic_after, -1, video_mean/pic_mean, 0);
          cv::imshow("pic_after", input_pic);
          
          input_pic.copyTo(roi);
      }
    }
    
    // 8. show mosaiced image to window
    cv::imshow("after", frame);
    cv::imshow("before", before);

    cv::VideoWriter output_video;
    output_video.open(OUT_VIDEO_FILE, CV_FOURCC('M','J','P','G'), 30, cv::Size(640, 480));
    output_video << frame;

    int key = cv::waitKey(10);
    if(key == 'q' || key == 'Q')
        break;
    }
 return 0;
}

std::tuple<cv::Mat, double> cuttingPicture(cv::Mat input_pic, cv::CascadeClassifier cascade, cv::CascadeClassifier cascade_eye){

  double scale = 4.0;
  cv::Mat gray, smallImg(cv::saturate_cast<int>(input_pic.rows/scale), 
                cv::saturate_cast<int>(input_pic.cols/scale), CV_8UC1);

  cv::cvtColor(input_pic, gray, CV_BGR2GRAY);
  cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
  cv::equalizeHist(smallImg, smallImg);
  std::vector<cv::Rect>  faces;
  std::vector<cv::Rect> eyes;

  cascade.detectMultiScale(smallImg, faces,
                    1.01,  //大きくするとモザイク範囲ちいさくなる(2.1)
                    2, //大きくするとモザイク判定がゆっくりになる(40ぐらいから)
                    CV_HAAR_SCALE_IMAGE,
                    cv::Size(30, 30));

  for(int i = 0; i < faces.size(); i++){
      cv::Point center;
      int radius;
      center.x = cv::saturate_cast<int>((faces[i].x + faces[i].width * 0.5) * scale);
      center.y = cv::saturate_cast<int>((faces[i].y + faces[i].height * 0.5) * scale);
      radius = cv::saturate_cast<int>((faces[i].width + faces[i].height) * 0.25 * scale);

      cv::Rect face_rect(center.x - radius, center.y - radius, radius * 2, radius * 2);
      cv::Mat faceRegion = input_pic(face_rect);
      cascade_eye.detectMultiScale(faceRegion, eyes,
                    1.1,  //大きくするとモザイク範囲ちいさくなる(2.1)
                    2, //大きくするとモザイク判定がゆっくりになる(40ぐらいから)
                    CV_HAAR_SCALE_IMAGE,
                    cv::Size(30, 30));
      double min_x = 1000000.0;
      double min_y = 1000000.0;
      double min_width = 1000000.0;
      double min_height = 1000000.0;
      for (auto eyeRect:eyes){
          if (min_x > eyeRect.x){
              min_x = eyeRect.x;
              min_y = eyeRect.y;
              min_width = eyeRect.width;
              min_height = eyeRect.height;
          }
          cv::Rect not_eye_rect(min_x, face_rect.y + min_y + min_height, face_rect.width, face_rect.height - min_height);

          cv::Rect center_rect(face_rect.x + min_x + min_width, face_rect.y + min_y, 10, min_height);
          cv::Mat center_pic = input_pic(center_rect).clone();
          double pic_mean = cv::mean(center_pic)[0];

          return {input_pic(not_eye_rect).clone(), pic_mean};
      }
  }
}