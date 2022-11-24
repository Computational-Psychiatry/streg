#ifndef MODEL_HPP
#define MODEL_HPP

class Model;

#include "RegFw.hpp"
#include "Utility.hpp"
#include "Image.hpp"
#include "GaborBank.hpp"
#include "MachineLearning.hpp"
#include <map>
#include <string>
#include <set>
#include <opencv2/core.hpp>


using std::vector; using std::cout; using std::endl;

class Label
{
public:

    //!!!
    Label(double _tx, double _ty, double _sc, double _r, double _absSc = -1) : tx(_tx), ty(_ty), sc(_sc), r(_r), absSc(_absSc) { if (_absSc > -1) absSc = absSc;}
    Label(const cv::Mat& mat) : tx(mat.at<double>(0,0)), ty(mat.at<double>(0,1)), sc(mat.at<double>(0,2)), r(mat.at<double>(0,3)) {}
    Label(int idx, const Model* m);
    double tx;
    double ty;
    double sc; // scale difference bw. pairs, i.e. sc1-sc2
    double r;

    double absSc; // new and optional, absolute scale

    friend Label operator-(const Label &lhs, const Label &rhs);

    Label combine(const Label& l);

    double avgErr(const cv::Mat &H_1, const cv::Mat &H_2) const;
    double avgErr() const;


    cv::Mat toMat() const;
    cv::Mat getWarpMatrix(const cv::Mat& im) const;
    void print() { cout << "AVG" << avgErr() << endl; }
    void print(const cv::Mat& H1,const cv::Mat& H2) { cout << "AVG" << avgErr(H1, H2) << endl; }
//    void print() { cout << "(" << std::setprecision(4) <<tx<<", "<<ty<<","<<sc << ", "<< r<< ")" << "AVG: " << avgErr() << endl; }
    Label invert() const { Label out(*this); out.tx = -out.tx; out.ty = -out.ty; out.sc = 1./out.sc; out.r = -out.r; return out; };
};

class Model
{
public:
    cv::Mat data;
    cv::Mat fullLabels;

    //! keys: label idxs, values: data row
    //! used for fetching all the rows of a certain label
    std::multimap<int,cv::Mat> dataMap;

    std::vector<int> labels;
    std::vector<double> maxs;
    std::vector<double> mins;
    std::vector<double> binCtrs;

    std::vector<cv::Mat> Ps;

    std::vector<int> samplesPerLabel;

    //! Limit the number of samples that can be used to train each label, so that one won't be biased over the other
    size_t numSamplesPerLabel;

    int refImWidth;

    // check if we need any more samples of this class
    bool hasEnoughSamplesOf(const Label& label) const { return samplesPerLabel[label2ix(label)] >= numSamplesPerLabel; }

    // check if we need any more samples of this class
    bool allLabelsFilled() const { return filledLabels.size() == NL; }

    size_t getNumLabels() const { return NL; }

    std::string getModelKey() const { return modelKey; }

    std::string getUniqueKey()  const { return std::string(modelKey+"|"+F->getUniqueKey()); }

    void increaseNumOfSamples(const Label& l);

    bool isLabelOutOfRange(const Label& l) const;

    // estimate label with Naive Bayes
    int estimateLabelNB(const std::vector<double>& feats) const;
    int estimateLabelNB(const cv::Mat& feats, double* probability = nullptr, double* convProbability = nullptr, double* totProbability = nullptr) const;
    int estimateLabelNB2(const cvip::Mat3& mat, double *probability = nullptr, double *convProbability = nullptr, double* totProbability = nullptr) const;

    double get_t() const { return t; }
    double get_sc() const { return sc; }
    double get_r() const { return r; }
    double get_dt() const { return dt; }
    double get_dsc() const { return dsc; }
    double get_dr() const { return dr; }

    std::string labExt() const { std::stringstream ss; ss << ".lab" << F->getTimeWindow(); return ss.str(); }

    //! label extension with the wildcard *
    std::string labExtW() const { std::stringstream ss; ss << "*.lab" << F->getTimeWindow(); return ss.str(); }

    Model(const std::string& featuresKey, const std::string& modelKey);
    Model(FeatureExtractor* const _Fptr, const std::string& modelKey, MultiMDN* mdnPtr);

    // keep a list of all labels whose number of samples are satisfied
    std::set<int> filledLabels;


    inline int findFeatureBinIdx(double rawFeatVal, int featIdx) const;
    inline int findFeatureBinIdx(const vector<double>& feats, int featIdx) const;
    inline int findFeatureBinIdx(const cv::Mat& feats, int featIdx) const;

    //! convert label_index to Label
    Label ix2label(int idx) const { return ix2labelTbl[idx]; }

    //! convert Label or (label) to label_index
    int label2ix(const Label& l) const { return label2ix(l.tx, l.ty, l.sc, l.r); }
    int label2ix(double _tx, double _ty, double _sc, double _r) const;

    //! Gabor bank
    GaborBank G;
    FeatureExtractor* F;

    MultiMDN* mdnPtr;

    //! strings for cache paths and identifiers
    std::string patchType; // which patch are we working on? Face, leye, reye, mouth?
    std::string featuresKey;
    std::string featuresPath;           // path of features of #featuresKey computed for all tuples
    std::string thresholdFeaturesPath;           // path of features of #featuresKey computed for all tuples
    std::string thresholdTestFeaturesPath;           // path of features of #featuresKey computed for all tuples
    std::string cacheDataPrePath;       // pre: feature cache of all tuples
    std::string cacheFullLabelsPrePath; // pre: label cache of all tuples
    std::string cacheDataPath;          // feature cache of tuples used in the model
    std::string cacheFullLabelsPath;    // label cache of tuples used in the model
    std::string cacheModelPath;         // cache dir of model (all likelihood functions)
    std::string cacheThresholdFilePath;

    double threshold; // threshold to approve registration confidence (loglikelihood)

private:
    //! operations common to all constructors
    void init();

    //! fairly generic function for finding nearest element in vector
    static long findNearestIdx(const std::vector<double>& vec, double val);

    //! called with constructor, construct space by the cartesian product of all labels
    void constructLabelSpace();

    //! number of bins for 1D likelihood functions
    size_t numBins;

    //! a minimum value to be added to each bin in a likelihood function
    double binEps;

    //! unique key to identify present model
    std::string modelKey;

    //! range and resolution of label spaces
    double t, dt, sc, dsc, r, dr;

    //! label index to label;
    std::vector<Label> ix2labelTbl;

    //! individual label spaces
    std::vector<double> txs, tys, scs, rs;

    //! size of individual label spaces
    size_t Ntx, Nty, Nsc, Nr;

    //! number of total labels
    size_t NL;

    //! Number of features
    size_t K;
};

using std::string; using std::initializer_list; using std::tuple;

class ModelsCascade
{
public:
    ModelsCascade(const std::string& featuresKey);

//    ModelsCascade(const vector< tuple<string, string> >& modelKeys);
    ModelsCascade(const vector<tuple<FeatureExtractor *, string, MultiMDN*> > &modelKeys);

    Model* modelForThreshold;

    int thresholdRefIdx;

    void train();

    void loadFromCache();

//    std::string getUniqueKey() const { return std::string(QString::number(models.size()).toStdString()+"-"+models.back().getUniqueKey()); }
    std::string getUniqueKey() const { return std::string(std::to_string(models.size())+"-"+models.back().getUniqueKey()); }

    std::vector<Model> models;

    double getThreshold() { return modelForThreshold->threshold; }

    std::string getPatchType() { return models.back().patchType; }

};





/**
 * @brief Model::findFeatureBinIdx
 * @param rawFeatVal
 * @param featIdx
 * @return
 */
int Model::findFeatureBinIdx(double rawFeatVal, int featIdx) const
{
    return findNearestIdx(binCtrs,(rawFeatVal-mins[featIdx])/(maxs[featIdx]-mins[featIdx]));
}

/**
 * @brief Model::findFeatureBinIdx
 * @param feats
 * @param featIdx
 * @return
 */
int Model::findFeatureBinIdx(const vector<double>& feats, int featIdx) const
{
    return findNearestIdx(binCtrs,(feats[featIdx]-mins[featIdx])/(maxs[featIdx]-mins[featIdx]));
}

/**
 * @brief Model::findFeatureBinIdx
 * @param feats
 * @param featIdx
 * @return
 */
int Model::findFeatureBinIdx(const cv::Mat& feats, int featIdx) const
{
    return findNearestIdx(binCtrs,(feats.at<double>(0,featIdx)-mins[featIdx])/(maxs[featIdx]-mins[featIdx]));
}







#endif // MODEL_HPP

