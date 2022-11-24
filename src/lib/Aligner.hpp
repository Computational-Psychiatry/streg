#ifndef ALIGNER_H
#define ALIGNER_H


//#define UPD_SELF_IMAGE 1 // update image I_tilde (rather than labels)



#include "Models.hpp"
#include "MachineLearning.hpp"
#include <deque>

using std::vector;
using cv::Mat;
using std::tuple;

class Aligner
{
public:

    int timeWindow;

    int Tw; // number of frames before correction is attempted

    int numIterations;

    int performedIterations;

    bool deletePointers;

    double operationScale;

    std::vector<FeatureExtractor*> Fptrs;
    std::vector<MultiMDN*> MDNptrs;


    Aligner(ModelsCascade* _MC, double _operationScale=1);

    Aligner(const std::string& patchType);

    ~Aligner();

    void init(ModelsCascade* _MC);


    //! from video after automatic face detection
    static std::vector<cv::Mat> fromVideo(const std::string &videoPath, bool useTrackerOutput = false);

    static std::vector<cv::Mat> fromFrames(const std::vector<cv::Mat>& frames, bool useTrackerOutput = false);

    static std::vector<cv::Point2f> locateLandmarks(const std::vector<cv::Mat>& frames);
    Label alignPair(const std::vector<cv::Mat>& _magicIms, int idx, double* convLogLikelihoodPtr);

    std::pair<bool, Label> attemptCorrection(int twrong, std::vector<cv::Mat>& magicIms, std::vector<Label> *estLabels, std::vector<Mat> *estWarps, std::vector<bool> *successFlag);


    //! create randomized CK sequences
    std::vector<std::vector<cv::Mat> > createPerturbedCKSequences(int startSubjectIdx, int numSubjects) const;

    static std::vector<cv::Mat> eliminateFalsePositives(const std::vector<cv::Mat>& frames, const std::vector<int>& clusterIdx);

    //! read images from folder to vector

    static std::vector<cv::Mat> readSeqClipFromVideo(const std::string& videopath, int tBegin, int tEnd, int frameSize);


    std::vector<cv::Mat> align(const std::vector<cv::Mat>& _magicIms, size_t ref, std::vector<cv::Mat> *beforeDiffs,
                                        std::vector<cv::Mat> *afterDiffs, std::vector<Label>* gtLabels, std::vector<Label>* estLabels, std::vector<cv::Mat> *Hs,
                                        std::vector<double> *convLogLikelihoods);


    std::tuple<std::vector<cv::Mat>, std::vector<int> > alignSequenceWithCorrection(const std::vector<cv::Mat>& _magicIms, std::vector<cv::Mat> *beforeDiffs,
                                                     std::vector<cv::Mat> *afterDiffs, std::vector<Label>* gtLabels, std::vector<Label>* estLabels,
                                                     std::vector<double> *convProbs, std::vector<cv::Mat> *estWarps, std::vector<bool> *successFlag, std::vector<cv::Mat> *allEstWarps=nullptr);


    //! align the last image in the segment 'magicIms'
    cv::Mat alignTmp(const std::vector<cv::Mat> &magicIms, int mIdx, double *convProb, Label *cumEst, size_t _MAX_IT = 0, bool greenLight = false, std::vector<cv::Mat> *allEstWarps=nullptr);

    //! align the last image in the segment 'magicIms'
    double computeConvergenceLikelihood(const std::vector<cv::Mat> &magicIms, const Model* modelPtr) const;

    std::pair<Label, bool> findReferenceToStichSegments(const std::vector<cv::Mat>& seg1,
                                                        const std::vector<cv::Mat> &seg2,
                                                        std::set<pair<int,int> >* pairsChecked = nullptr, int offsetSeg1=0, int offsetSeg2=0, int *idxSeg1 = nullptr, int *idxSeg2 = nullptr);

    std::vector<cv::Mat> stitchAllParts(const std::vector<std::vector<cv::Mat> >& parts, const std::vector<std::vector<bool> > &successFlags);


    std::vector<std::vector<cv::Mat> > stichNeighbouringSegments(std::vector<std::vector<cv::Mat> >& segments, std::vector<Label> &labels, std::set<std::pair<int, int> > *checkedPairs = nullptr);

    static std::vector<cv::Mat> getRegisteredSequence(const std::vector<cv::Mat>& magicIms, std::vector<Label> &estLabels);

    tuple<vector<Mat>, vector<bool>, vector<Mat> > alignOnline(const std::vector<cv::Mat>& _magicIms, bool identifyFailure=false);


//    std::vector<cv::Mat> alignSequence(const std::vector<cv::Mat>& magicIms, std::vector<Label> *estLabels, std::vector<int> &clusterIndices, std::vector<cv::Mat> *diffsBefore = nullptr, std::vector<cv::Mat> *diffsAfter = nullptr) const;
    std::vector<cv::Mat> alignSequence(const std::vector<cv::Mat>& magicIms, std::vector<cv::Mat>& outMagicIms,
                                       std::vector<Label> *estLabels, std::vector<int> &registrationStatus,
                                       std::vector<cv::Mat> *diffsBefore = nullptr, std::vector<cv::Mat> *diffsAfter = nullptr,
                                       std::vector<cv::Mat> *estWarps = nullptr, std::vector<bool> *successFlag = nullptr,
                                       std::vector<cv::Mat> *allEstWarps = nullptr);


    std::pair<std::vector<cv::Mat>, std::vector<int> > stichDistantSegments(std::vector<std::vector<cv::Mat> >& segments, std::vector<Label> &estLabels, std::set<std::pair<int, int> > *checkedPairs = nullptr);

    //! align pair (independently from any sequence, do full aligning)
    std::pair<Label, bool> alignPair(std::tuple<cv::Mat, cv::Mat> &pair, double *convLogLikelihoodPtr = nullptr, int idx = -1);

    //! identify a possible oscillation in the process of registration (i.e. are we going back and forth between two estimations?
    bool isOscillating(const std::vector<size_t>& labels, size_t allowedRepetitions = 2) const;

    //! find a reference image to register other images to
    size_t findReferenceImage(const std::vector<cv::Mat>& refIm, const std::vector<Label> *rawLabels = nullptr) const;

    template <class T>
    static int getSegmentOffset(const std::vector<std::vector<T> >& vec, int idx);

    //! save the output as before-after videos to the "out" folder
    void saveOutput(const std::string &db, const std::string& patchType, size_t i,
                    vector<Mat> &before, vector<Mat> &after, vector<Mat> *dbefore = nullptr,
                    vector<Mat> *dafter = nullptr, std::vector<cv::Point2f> *landmarks = nullptr, std::vector<cv::Mat>* estWarps = nullptr,
                    std::vector<bool>* successFlag = nullptr, std::vector<int> *registrationStatus = nullptr, const std::string &extraText = "") const;


    void saveRegisteredVideo(const std::string &path, const string &patchType,
                    vector<Mat> &before, vector<Mat> &after, std::vector<cv::Mat>* estWarps = nullptr,
                             std::vector<int> *registrationStatus = nullptr, std::vector<std::vector<double> > *poses = nullptr, std::vector<cv::Mat> *landmarks = nullptr,
                             bool saveBefore = false, const std::string &vidOutPath = "", const std::string &successOutPath = "", int FPS=30, const std::string &vidBeforeOutPath = "") const;

    ModelsCascade* MC;
    MLP* thresholdMlp;
    FeatureExtractor* thresholdF;
};

#endif // ALIGNER_H
