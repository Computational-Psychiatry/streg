#include "FeatureExtractor.hpp"

FeatureExtractor::FeatureExtractor()
{
}

std::vector<double> FeatureExtractor::pool(const cv::Mat &en, std::string curFeatureType, size_t curNumParts)
{
    std::vector<double> features;
    size_t w = en.cols;
    size_t h = en.rows;

    // part width, part height
    size_t pw = std::floor((double)w/curNumParts);
    size_t ph = std::floor((double)h/curNumParts);

    for (size_t y=0; y<curNumParts; ++y)
    {
        for (size_t x=0; x<curNumParts; ++x)
        {
            double mn, mx;
            cv::Scalar mean,stdv;
            cv::Mat part = en(cv::Rect(x*pw,y*ph,pw,ph)).clone();
            cv::meanStdDev(part, mean, stdv);
            cv::minMaxLoc(part, &mn, &mx);

            //features.push_back(mx);
            if (curFeatureType == "meanstd") {
                features.push_back(mean[0]);
                features.push_back(stdv[0]);
            } else if (curFeatureType == "std") {
                features.push_back(stdv[0]);
            } else if (curFeatureType == "max") {
                features.push_back(mx);
            } else if (curFeatureType == "mean") {
                features.push_back(mean[0]);
            }
        } //std::vector<double>{1.,1./2,1./4,1./6,1./8});
    }

    return features;
}


double FeatureExtractor::signalEnergy(const std::vector<double> &x) const
{
    double energy = 0;

    for (size_t i=0; i<x.size(); ++i)
    {
        double tmp = x[i];
        energy += tmp*tmp;
    }

    return energy;
}
