#include "OpticalFlow.hpp"
#include "Utility.hpp"
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef GUI
#endif


OpticalFlow::OpticalFlow(const std::string& featuresKey)
    : timeWindow(2),
      uniqueKey(featuresKey)
{
    std::vector<std::string> l = split(featuresKey, '-');

    size_t cnt = 0;
    std::string dummy = l[cnt++].c_str();

    numParts = std::atoi(l[cnt++].c_str());
    windowSize = std::atoi(l[cnt++].c_str());

    std::vector<cv::Mat> dummyMats;

    for (size_t i=0; i<timeWindow; ++i)
        dummyMats.push_back(cv::Mat(200,200,CV_8U));

    std::vector<double> dummyFeats = computeFeatures(dummyMats);

    vectorSize = dummyFeats.size();
}

void OpticalFlow::drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, double, const cv::Scalar& color)
 {
    using namespace cv;
     for(int y = 0; y < cflowmap.rows; y += step)
         for(int x = 0; x < cflowmap.cols; x += step)
         {
             const Point2f& fxy = flow.at<Point2f>(y, x);
             line(cflowmap, Point(x,y), Point(cvRound(x+8*fxy.x), cvRound(y+8*fxy.y)),
                  color);
             circle(cflowmap, Point(x,y), 2, color, -1);
         }
}

void FeatureExtractor::drawMap(const std::vector<int>& fts)
{
    int D = fts.size()/2;

    cv::Mat map(200,200, CV_8UC3, cv::Scalar::all(0));

    int w=20;

    for (size_t i=0; i<D; ++i)
    {
        int x = fts[i]%10;
        int y  = fts[i]/10;

        int xS = x*w;
        int xE = xS+w;

        int yS = y*w;
        int yE = yS+w;

        cv::rectangle(map, cv::Rect(xS, yS, xE-xS, yE-yS), cv::Scalar(0,255,0),-1,0);
    }

#ifdef GUI
    cv::imshow("map", map);
#endif
}




std::vector<double> OpticalFlow::computeFeatures(const cvip::Mat3 &mat) const
{
    std::vector<cv::Mat> ims = mat.frames;
    cv::Mat im1 = ims[0];
    cv::Mat im2 = ims[1];

    cv::Mat flow;

    cv::calcOpticalFlowFarneback(im1, im2, flow, 0.5, 3, windowSize, 3, 5, 1.2, 0);

    std::vector<cv::Mat> channels;
    cv::split(flow, channels);

    cv::Mat cflowmap(200,200,CV_8UC3, cv::Scalar::all(0));

    /*
    drawOptFlowMap(flow, cflowmap, 6, 1, cv::Scalar(0,255,0));
    cv::imshow("asdasd", cflowmap);
    cv::waitKey(20);

//    cv::imshow("fla", cvip::Image::doubleToUchar(channels[0]));
//    cv::imshow("flb", cvip::Image::doubleToUchar(channels[1]));
//    cv::waitKey(0);
    */

    std::vector<double> featsX = pool(channels[0], "mean", numParts);
    std::vector<double> featsY = pool(channels[1], "mean", numParts);

    std::vector<double> out;
    for (size_t i=0; i<featsX.size(); ++i)
        out.push_back(featsX[i]);

    for (size_t i=0; i<featsY.size(); ++i)
        out.push_back(featsY[i]);

    return out;
}
