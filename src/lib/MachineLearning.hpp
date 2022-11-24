#ifndef MACHINELEARNING_H
#define MACHINELEARNING_H

#include "Image.hpp"



//! Mixture Density Network
class MDN
{
public:
    MDN() {}
    MDN(const std::string& modelPath) { init(modelPath); }

    //! initialize from files
    void init(const std::string& modelPath);

    //! forward propagation
    std::vector<double> forward(const cv::Mat& x) const;

    //! predict -- regression
    std::vector<double> predict(const cv::Mat& x, double* prior = nullptr) const;

private:

    cv::Mat W1;
    cv::Mat W2;
    cv::Mat b1;
    cv::Mat b2;
    cv::Mat mean;
    cv::Mat std;

    int K;          // number of mixture components
    int D;          // number of features
    int Nhidden;    // number of weights
    int Nout;       // dimensionality of network output -- includes everything, mixture coeff etc.
    int Ntarget;    // dimensionality of final output; i.e. length of misalignment vector (4 for Euclidean, 6 for affine)

};

//! Mixture Density Network
class MLP
{
public:
    MLP() {}
    MLP(const std::string& modelPath) { init(modelPath); }

    //! initialize from files
    void init(const std::string& modelPath);

    //! forward propagation
    std::vector<double> forward(const cv::Mat& x) const;

    //! predict -- regression
    std::vector<double> predict(const cv::Mat& x, double* prior = nullptr) const;

    //! sigmoid
    double sigmoid(double x) const;

    bool converged(const cv::Mat& x) const;

    std::vector<int> selectedFeats;

    double threshold;

private:

    cv::Mat W1;
    cv::Mat W2;
    cv::Mat b1;
    cv::Mat b2;

    int D;          // number of features
    int Nhidden;    // number of weights
    int Nout;       // dimensionality of network output -- includes everything, mixture coeff etc.
    int Ntarget;    // dimensionality of final output; i.e. length of misalignment vector (4 for Euclidean, 6 for affine)


};







class MultiMDN
{
public:
    std::vector<MDN> nets;
    std::vector<double> centres;

    MultiMDN() {}
    MultiMDN(const std::string& modelPath) { init(modelPath); }

    //! initialize from files
    void init(const std::string& modelPath);

    //! predict -- regression
    std::vector<double> predict(const cv::Mat& x, double energy, double* prior = nullptr, int *clusterIdxOutput = nullptr) const;

    static long findNearestIdx(const std::vector<double>& vec, double val);

    //! number of centres in mixture model
    int K;

};






#endif // MACHINELEARNING_H
