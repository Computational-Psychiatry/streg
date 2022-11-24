#ifndef GABORBANK_HPP
#define GABORBANK_HPP

#include "RegFw.hpp"
#include "Utility.hpp"
#include "Image.hpp"
#include "FeatureExtractor.hpp"
#include <string>
#include <bitset>

#include <opencv2/core.hpp>


using namespace cvip;

using std::tuple; using cv::Mat; using std::pair; using std::string;


class GaborBank : public FeatureExtractorHelper<GaborBank>
{
public:
    GaborBank(size_t _numPartsX=2,
              size_t _numPartsY=2,
              const std::string& _featureType = "mean",
              const std::string& _normalisationType = "single",
              int _numScales = 5,
              const std::string &_scalesBitset = "11111",
              const std::string &_orientsBitset = "11111111");

    GaborBank(const std::string& str);

    //! A string that will identify the present Gabor filter combination
    std::string getUniqueKey() const { return uniqueKey; }

    //! compute motion energies
    static std::vector<cv::Mat> computeEnergies(tuple<Mat,Mat>& tup1,
                                                const std::vector<double> &_scales,
                                                const std::vector<tuple<Mat,Mat> > &_reFilters,
                                                const std::vector<tuple<Mat,Mat> > &_imFilters);

    //! compute features motion energy for an image pair
    std::vector<double> computeFeatures(const Mat3 &mat) const;

    double signalEnergy(const std::vector<double> &representation) const;


    size_t getTimeWindow() const { return timeWindow; }

    //! compute features motion energy for an image pair
    /**
    ****************************
    static  std::map<string, std::vector<double> > computeFeatures(tuple<Mat,Mat> tup,
                                                           const vector<double>& _scales,
                                                           const vector<double>& _orients,
                                                           const vector<tuple<Mat,Mat> >& _reFilters,
                                                           const vector<tuple<Mat,Mat> >& _imFilters,
                                                           const vector<string>& featureTypes,
                                                           const vector<string>& normalisationTypes,
                                                           const vector<size_t> &numPartss,
                                                           const std::string& scalesBitsetStr,
                                                           const std::string& orientsBitsetStr);

                                                           **************************************/
    enum T {T1, T2};

    std::vector<double> getScales() const { return scales; }
    std::vector<double> getOrients() const { return orients; }

    std::bitset<5> getScalesBitset() const { return scalesBitset; }
    std::bitset<8> getOrientsBitset() const { return orientsBitset; }

    size_t numFeatures() const { return numPartsX*numPartsY*orients.size()*scales.size()*((featureType == "meanstd" ? 2 : 1)); }

    static size_t computeNumFeatures(const std::string& _featureType,
                                     size_t _numParts,
                                     const std::vector<double>& orients,
                                     const std::vector<double>& scales);

    static std::string computeUniqueKey(const string& _featureType,
                                        const string& _normalisationType,
                                        const size_t _timeWindow,
                                        const size_t _numParts,
                                        const string &_scalesBitsetStr,
                                        const string &_orientsBitsetStr);

    //! compute features from motion energies
    /**
    static std::vector<double> _computeFeatures(const std::vector<cv::Mat>& energies,  const std::string& _featureType, size_t _numParts);
    **/

    //! compute features straight from pairs
    std::vector<double> _computeFeatures(const Mat3 &mat3d) const;

    static void extractSlice( const cv::Mat& image3d, const int z, cv::Mat& slice );

    size_t getNumScales() const { return numScales; }


    //! real and imaginary parts of each filter
    std::vector<tuple<Mat,Mat> > reFilters;
    std::vector<tuple<Mat,Mat> > imFilters;

    //! real and imaginary parts of each filter
    std::vector<std::vector<cv::Mat> > reFilters2;
    std::vector<std::vector<cv::Mat> > imFilters2;

private:



    /*
    //! compute Gabor features
    std::vector<double> _computeFeatures(tuple<Mat,Mat>& tup1, bool squareUp,
                                         std::vector<double> lambda1s = std::vector<double>(),
                                         std::vector<double> lambda2s = std::vector<double>()) const;
    */

    //! create the filter bank
    void createFilters();

    //! number of parts to pool gabor motion energy
    size_t numPartsX;
    size_t numPartsY;

    //! the time window (originally was 2 -- pairs --, now more general)
    size_t timeWindow;

    //! how many scales are going to be used?
    size_t numScales;

    //! bitsets that decide which scales and orientations will be used by this filter bank
    std::bitset<5> scalesBitset;
    std::bitset<8> orientsBitset;

    //! normalisation type; use a single image or two images
    std::string normalisationType;

    //! feature type; mean, max or std
    std::string featureType;

    //! scales to apply the filters
    std::vector<double> scales;

    //! orientations to apply the filters
    std::vector<double> orients;

    //! a string that uniquely identifies the gabor bank configuration (num parts,scales etc.)
    std::string uniqueKey;

    //! create a filter pair where first filter is filter of t=1 and second t=2
    //! (both are either real or imag, depending on phi)
    tuple<Mat,Mat> createFilterTuple(int w, int h, double theta, double v, double phi);

    //! create a filter pair where first filter is filter of t=1 and second t=2
    //! (both are either real or imag, depending on phi)
    std::vector<cv::Mat> createFilterTuple(int w, int h, size_t T, double theta, double v, double phi);

};

#endif // GABORBANK_HPP
