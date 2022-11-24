#ifndef FEATUREEXTRACTOR_HPP
#define FEATUREEXTRACTOR_HPP

#include <string>
#include "Image.hpp"

class FeatureExtractor
{
public:
    FeatureExtractor();

    virtual std::string getUniqueKey() const = 0;
    virtual size_t numFeatures() const = 0;
    virtual std::vector<double> computeFeatures(const cvip::Mat3& mat) const = 0;
    virtual size_t getTimeWindow() const = 0;
    virtual FeatureExtractor* clone() const = 0;
    virtual double signalEnergy(const std::vector<double>& representation) const;

    template <typename T> static int sgn(T val) { return (T(0) < val) - (val < T(0)); }
    static void drawMap(const std::vector<int>& fts);

    //! pool over subregions; compute mean and standard deviation for subregions
    static std::vector<double> pool(const cv::Mat& region, std::string curFeatureType, size_t curNumParts);

};

template <class Derived>
class FeatureExtractorHelper : public FeatureExtractor
{
public:
  virtual FeatureExtractor* clone() const
  {
    return new Derived(static_cast<const Derived&>(*this)); // call the copy ctor.
  }
};


#endif // FEATUREEXTRACTOR_HPP

