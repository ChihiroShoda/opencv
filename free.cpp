#include <opencv2/opencv.hpp>
#include <stdio.h>

#define OUT_VIDEO_FILE "sample_video_output.avi"
const char *preset_file = "nose_and_mouth.JPG";

int size_of_mosaic = 0;

int main(int argc, char *argv[])
{
  // 1. load classifier
  std::string cascadeName = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"; //Haar-like
  std::string cascadeName_eye = "/usr/share/opencv/haarcascades/haarcascade_eye.xml";
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
  cv::VideoCapture cap;
  cap.open(0);
  cap >> frame;

  //鼻と口の画像
  cv::Mat input;
  input = cv::imread(preset_file, 1);
  if(input.empty()){
      fprintf(stderr, "cannot open %s\n", preset_file);
      exit(0);
  }

  // 3. prepare window and trackbar
  cv::namedWindow("result", 1);
  cv::createTrackbar("size", "result", &size_of_mosaic, 30, 0);

  double scale = 4.0;
  cv::Mat gray, smallImg(cv::saturate_cast<int>(frame.rows/scale), 
                cv::saturate_cast<int>(frame.cols/scale), CV_8UC1);

  for(;;){

    // 4. capture frame
    cap >> frame;
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
      if(size_of_mosaic < 1) size_of_mosaic = 1;
      cv::Rect roi_rect(center.x - radius, center.y - radius, radius * 2, radius);
      double face_width = roi_rect.width;
      cv::Mat faceRegion = frame(roi_rect);
      //cv::rectangle(frame, roi_rect, cv::Scalar(0, 0, 255), 2);
      cascade_eye.detectMultiScale(faceRegion, eyes,
                    1.1,  //大きくするとモザイク範囲ちいさくなる(2.1)
                    2, //大きくするとモザイク判定がゆっくりになる(40ぐらいから)
                    CV_HAAR_SCALE_IMAGE,
                    cv::Size(30, 30));
      double min_x = 1000000.0;
      double min_y = 1000000.0;
      double min_height = 1000000.0;
      for (auto eyeRect:eyes){
          cv::Rect eye_rect(roi_rect.x + eyeRect.x, roi_rect.y + eyeRect.y, eyeRect.width, eyeRect.height);
          //cv::rectangle(frame, eye_rect, cv::Scalar(0, 255, 0), 2);
          if (min_x > eyeRect.x){
              min_x = eyeRect.x;
              min_y = eyeRect.y;
              min_height = eyeRect.height;
          }
          cv::resize(input, input, cv::Size(), face_width/input.cols, face_width/input.cols, cv::INTER_LINEAR);
          cv::Mat roi = frame(cv::Rect(roi_rect.x, roi_rect.y + min_y + min_height, input.cols, input.rows));
          input.copyTo(roi);
      }

      //cv::resize(input, input, faceRegion.size(), 0, 0, cv::INTER_LINEAR);
      //CV_Assert(())
      //cv::Mat tmp;
      //cv::bitwise_not(mosaic,mosaic);
      //cv::resize(mosaic, tmp, cv::Size(radius / size_of_mosaic, radius / size_of_mosaic), 0, 0);
      //cv::resize(tmp, mosaic, cv::Size(radius * 2, radius * 2), 0, 0, CV_INTER_NN);
    }
    

    // 8. show mosaiced image to window
    cv::imshow("result", frame);

    cv::VideoWriter output_video;
    output_video.open(OUT_VIDEO_FILE, CV_FOURCC('M','J','P','G'), 30, cv::Size(640, 480));
    output_video << frame;

    int key = cv::waitKey(10);
    if(key == 'q' || key == 'Q')
        break;
    }
 return 0;
}
