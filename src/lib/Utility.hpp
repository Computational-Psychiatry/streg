#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <vector>
#include "RegFw.hpp"
#include "Image.hpp"

class Label;

namespace imutil
{

//! crop the image from the center
cv::Mat cropCore(const cv::Mat &im, int refImWidth = 200, double *shiftBy=nullptr);

//! slow version of cropCore -- about to be deprecated
cv::Mat cropCoreOld(const cv::Mat &im, int refImWidth = 200);

//! resize the image (after upscaling it)
cv::Mat resizeUpscaled(const cv::Mat& im, double r);

//! circular shift for translation
cv::Mat shift(const cv::Mat& im, double tx, double ty);

//! resize the image (after upscaling it)
void resizeUpscaledSelf(cv::Mat& im, double r);

//! circular shift for translation
void shiftSelf(cv::Mat& im, double tx, double ty);

//! rotate (after upscaling)
void rotateSelf(cv::Mat& im, double theta);

//! rotate and return new image (after upscaling)
cv::Mat rotate(const cv::Mat& im, double theta);

//! inner function for circular shift
cv::Mat shiftRows(cv::Mat& im, int t);

cvip::Rect rectFromLandmarks(const cv::Mat& pts, const std::string &patchType, const cv::Size sz);

cv::Mat createCanonicalFace(const cv::Mat frame, const cv::Mat& pts);

cv::Point2f pointFromLandmarks(const cv::Mat& pts, const std::string &patchType);

cv::Mat applyLabel(cv::Mat& im, const Label &l, double operationScale=1);
cv::Mat applyLabel(cv::Mat& im, const cv::Mat &H);
void applyLabelReset(cv::Mat& im, const Label &l);

void applyLabelOld(cv::Mat& im, Label& l);
void applyLabelOldOld(cv::Mat& im, Label& l);

cv::Point2f avgPt(const std::vector<cv::Point2f> &pts);



}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &s, char delim);


#endif // UTILITY_HPP
