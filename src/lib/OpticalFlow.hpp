#ifndef OPTICALFLOW_HPP
#define OPTICALFLOW_HPP

#include "Image.hpp"
#include "FeatureExtractor.hpp"
#include <bitset>

class OpticalFlow: public FeatureExtractorHelper<OpticalFlow>
{
public:
    OpticalFlow(const std::string& featureKey);


    int timeWindow;
    int vectorSize;
    int numParts;
    int windowSize;
    std::string uniqueKey;

    static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, double, const cv::Scalar& color);

    size_t getTimeWindow() const { return timeWindow; }

    size_t numFeatures() const { return vectorSize; }

    std::string getUniqueKey() const { return uniqueKey; }

//    double signalEnergy(const std::vector<double> &representation) const;

    std::vector<double> computeFeatures(const cvip::Mat3 &mat) const;


};

#endif // OPTICALFLOW_HPP
