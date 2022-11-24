#include "Models.hpp"
#include <chrono>
#include <queue>
#include <numeric>
#include <map>

/**
 * @brief Label::Label - create Label from label index
 * @param idx - label index
 * @param m - Model
 */
Label::Label(int idx, const Model *m)
{
    *this = m->ix2label(idx);
}


Label operator-(const Label &lhs, const Label &rhs) {
    return Label(lhs.tx-rhs.tx, lhs.ty-rhs.ty,lhs.sc-rhs.sc,lhs.r-rhs.r);
}



Label Label::combine(const Label &l)
{
    Label out(0,0,1,0);
    out.tx = tx+l.tx;
    out.ty = ty+l.ty;
    out.sc = sc*l.sc;
    out.r = r+l.r;

    return out;
}

double Label::avgErr() const
{
    double rad = r*cvip::PI/180.;
    double st = std::sin(rad);
    double ct = std::cos(rad);

    double s = sc+1;

    cv::Mat H(3,3,CV_64FC1, cv::Scalar::all(0));

    H.at<double>(0,0) = s*ct;
    H.at<double>(0,1) = -s*st;
    H.at<double>(0,2) = tx;
    H.at<double>(1,0) = s*st;
    H.at<double>(1,1) = s*ct;
    H.at<double>(1,2) = ty;
    H.at<double>(2,0) = 0;
    H.at<double>(2,1) = 0;
    H.at<double>(2,2) = 1;

    cv::Mat p1(3,1,CV_64FC1, cv::Scalar::all(0));
    cv::Mat p2(3,1,CV_64FC1, cv::Scalar::all(0));
    p1.at<double>(0,0) = 0;
    p1.at<double>(1,0) = 100;
    p1.at<double>(2,0) = 1;

    p2.at<double>(0,0) = 200;
    p2.at<double>(1,0) = 100;
    p2.at<double>(2,0) = 1;

    cv::Mat np1(3,1,CV_64FC1, cv::Scalar::all(0));
    cv::Mat np2(3,1,CV_64FC1, cv::Scalar::all(0));

    np1 = H*p1;
    np2 = H*p2;

    double err11 = p1.at<double>(0,0)-np1.at<double>(0,0);
    double err12 = p1.at<double>(1,0)-np1.at<double>(1,0);
    double err21 = p2.at<double>(0,0)-np2.at<double>(0,0);
    double err22 = p2.at<double>(1,0)-np2.at<double>(1,0);

    double err1 = err11*err11+err12*err12;
    double err2 = err21*err21+err22*err22;
/****
    std::cout << p1.at<double>(0,0) << std::endl;
    std::cout << p1.at<double>(0,1) << std::endl;
    std::cout << p2.at<double>(0,0) << std::endl;
    std::cout << p2.at<double>(0,1) << std::endl;

    std::cout << np1.at<double>(0,0) << std::endl;
    std::cout << np1.at<double>(0,1) << std::endl;
    std::cout << np2.at<double>(0,0) << std::endl;
    std::cout << np2.at<double>(0,1) << std::endl;
****/
    double err = std::sqrt(err1)+std::sqrt(err2);
    return err/2.;

}

double Label::avgErr(const cv::Mat& H_1, const cv::Mat& H_2) const
{

    cv::Mat bot = cv::Mat::zeros(1,3,CV_64FC1);
    bot.at<double>(0,2) = 1;

    cv::Mat H1 = H_1.clone();
//    H1.push_back(bot);
    cv::Mat H2 = H_2.clone();
    H2.push_back(bot);


    cv::Mat p1(3,1,CV_64FC1, cv::Scalar::all(0));
    cv::Mat p2(3,1,CV_64FC1, cv::Scalar::all(0));
    p1.at<double>(0,0) = 0;
    p1.at<double>(1,0) = 100;
    p1.at<double>(2,0) = 1;

    p2.at<double>(0,0) = 200;
    p2.at<double>(1,0) = 100;
    p2.at<double>(2,0) = 1;

    cv::Mat np1(3,1,CV_64FC1, cv::Scalar::all(0));
    cv::Mat np2(3,1,CV_64FC1, cv::Scalar::all(0));

    np1 = H2*p1;
    np2 = H2*p2;

    p1 = H1*p1;
    p2 = H1*p2;

    double err11 = p1.at<double>(0,0)-np1.at<double>(0,0);
    double err12 = p1.at<double>(1,0)-np1.at<double>(1,0);
    double err21 = p2.at<double>(0,0)-np2.at<double>(0,0);
    double err22 = p2.at<double>(1,0)-np2.at<double>(1,0);

    double err1 = err11*err11+err12*err12;
    double err2 = err21*err21+err22*err22;
/****
    std::cout << p1.at<double>(0,0) << std::endl;
    std::cout << p1.at<double>(0,1) << std::endl;
    std::cout << p2.at<double>(0,0) << std::endl;
    std::cout << p2.at<double>(0,1) << std::endl;

    std::cout << np1.at<double>(0,0) << std::endl;
    std::cout << np1.at<double>(0,1) << std::endl;
    std::cout << np2.at<double>(0,0) << std::endl;
    std::cout << np2.at<double>(0,1) << std::endl;
****/
    double err = std::sqrt(err1)+std::sqrt(err2);
    return err/2.;

}

cv::Mat Label::toMat() const
{
    double data[] = {tx, ty, sc, r};
    cv::Mat mat = Mat(1, 4, CV_64F, data).clone();
    return mat;
}

cv::Mat Label::getWarpMatrix(const cv::Mat &im) const
{
    int64 t1 = cv::getTickCount();

    cv::Mat trans = cv::Mat::zeros(2,3,CV_64F);
    trans.at<double>(0,0) = 1./sc;
    trans.at<double>(1,1) = 1./sc;

    trans.at<double>(0,2) = -tx;
    trans.at<double>(1,2) = -ty;

    cv::Mat tmp = (cv::Mat_<double>(1,3)<< 0., 0., 1);
    trans.push_back(tmp);


    double angle = -r;

    cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(im.cols/2.,im.rows/2.), angle, 1.);
    cv::Mat H = rot*trans;

    return H;
}

/**
 * @brief Model::Model
 * @param _featuresKey - a string that identifies the GaborBank combination
 * @param _modelKey - the string to read model params from
 */
Model::Model(const std::string& _featuresKey, const std::string &_modelKey)
    : featuresKey(_featuresKey),
      modelKey(_modelKey),
      G(_featuresKey)
{
    std::vector<std::string> strings = split(modelKey, '-');

    size_t tmp=0;

    patchType = strings[tmp++];
    numBins = std::atof(strings[tmp++].c_str());
    binEps = std::atof(strings[tmp++].c_str());
    t = std::atof(strings[tmp++].c_str());
    dt = std::atof(strings[tmp++].c_str());
    sc = std::atof(strings[tmp++].c_str());
    dsc = std::atof(strings[tmp++].c_str());
    r = std::atof(strings[tmp++].c_str())+rfw::EPS;
    dr = std::atof(strings[tmp++].c_str());
    numSamplesPerLabel = std::atoi(strings[tmp++].c_str());

    init();
}


/**
 * @brief Model::Model
 * @param _featuresKey - a string that identifies the GaborBank combination
 * @param _modelKey - the string to read model params from
 */
Model::Model(FeatureExtractor* const _Fptr, const std::string &_modelKey, MultiMDN *_mdnPtr)
    : modelKey(_modelKey),
      F(_Fptr),
      featuresKey(F->getUniqueKey()),
      mdnPtr(_mdnPtr)
{
    std::vector<std::string> strings = split(modelKey, '-');

    size_t tmp=0;

    patchType = strings[tmp++];
    numBins = std::atof(strings[tmp++].c_str());
    binEps = std::atof(strings[tmp++].c_str());
    t = std::atof(strings[tmp++].c_str());
    dt = std::atof(strings[tmp++].c_str());
    sc = std::atof(strings[tmp++].c_str());
    dsc = std::atof(strings[tmp++].c_str());
    r = std::atof(strings[tmp++].c_str())+rfw::EPS;
    dr = std::atof(strings[tmp++].c_str());
    numSamplesPerLabel = std::atoi(strings[tmp++].c_str());


    init();
}


/**
 * @brief Model::init operations common to all constructors
 */
void Model::init()
{
    //featuresPath = "data/samples/"+featuresKey+"/stra";
    featuresPath = "data/samples/"+featuresKey+"/"+patchType;
    thresholdFeaturesPath = "data/samples/"+featuresKey+"/threshold_"+patchType;
    thresholdTestFeaturesPath = rfw::TEST_SAMPLES_FOR_THRESHOLD_PATH +"/"+getUniqueKey();
    cacheDataPath = "data/cache/model_"+featuresKey+"/data_"+modelKey+".dat";
    cacheFullLabelsPath = "data/cache/model_"+featuresKey+"/fullLabels_"+modelKey+".dat";
    cacheDataPrePath = "data/cache/model_"+featuresKey+"/dataPreCache_" + patchType +".mat";
    cacheFullLabelsPrePath = "data/cache/model_"+featuresKey+"/fullLabelsPreCache_" + patchType + ".mat";
    cacheThresholdFilePath = "data/cache/model_"+featuresKey+"/threshold_" + patchType +".mat";

    std::string tmp = "data/cache/model_"+featuresKey;


    cacheModelPath = tmp+"/"+modelKey;

    //?K = G.numFeatures();
    K = F->numFeatures();
    constructLabelSpace();

    for (size_t i=0; i<NL; ++i)
    {
        samplesPerLabel.push_back(0);
    }

    if (patchType == "face") {
        refImWidth = 200;
    } else {
        refImWidth = 70;
    }

    double dbin = 1./numBins;
    for (size_t i=0; i<numBins; ++i) {
        binCtrs.push_back(dbin*i+dbin/2.);
    }
}

/**
 * @brief Model::findNearestIdx - generic function for finding the index of the nearest double in a vector
 * @param vec - vector to search in
 * @param val - value to match
 * @return
 */
long Model::findNearestIdx(const std::vector<double>& vec, double val)
{
    auto _x = min_element(begin(vec), end(vec), [=] (double x, double y) {
        return fabs(x-val) < fabs(y-val);
    });

    long x = std::distance(begin(vec),_x);
}

/**
 * @brief Model::label2ix - convert label to label index
 * @param ltx
 * @param lty
 * @param lsc
 * @return
 */
int Model::label2ix(double ltx, double lty, double lsc, double lr) const
{
    auto itx = findNearestIdx(txs, ltx);
    auto ity = findNearestIdx(tys, lty);
    auto isc = findNearestIdx(scs, lsc);
    auto ir = findNearestIdx(rs, lr);

    return ir*Nty*Ntx*Nsc+itx*Nty*Nsc+Nsc*ity+isc;
}

/**
 * @brief Model::constructLabelSpace
 */
void Model::constructLabelSpace()
{
    for (double x = -t; x<=t; x+=dt) {
        txs.push_back(x);
        tys.push_back(x);
        if (dt == 0)
            break; // avoid infinite loop
    }
    Ntx = txs.size();
    Nty = tys.size();

    for (double x = -sc; x<=sc; x+=dsc) {
        scs.push_back(x);
        if (dsc == 0)
            break; // avoid infinite loop
    }
    Nsc = scs.size();

    for (double x = -r; x<=r; x+=dr) {
        rs.push_back(x);
        if (dr == 0)
            break; // avoid infinite loop
    }
    Nr = rs.size();
    NL = Nsc*Ntx*Nty*Nr;

    size_t idx = 0;
    for (size_t p=0; p<Nr; ++p) {
        for (size_t i=0; i<Ntx; ++i) {
            for (size_t j=0; j<Nty; ++j) {
                for (size_t k=0; k<Nsc; ++k) {
                    size_t l = p*Ntx*Nty*Nsc+i*Nty*Nsc+Nsc*j+k;
                    ix2labelTbl.push_back(Label(txs[i],tys[j],scs[k],rs[p]));
                    idx++;
                }
            }
        }
    }
}



int Model::estimateLabelNB(const cv::Mat &feats, double* probability, double* convProbability, double *totProbability) const
{
    using namespace std;
    typedef pair<int,double> Pr;

    //! create a priority queue which will be giving you argmax efficiently
    auto comp = [](Pr p1, Pr p2){return p2.second>p1.second;};
    priority_queue<Pr, vector<Pr>, decltype(comp)> pq(comp);

    std::vector<double> logProbs(NL,0.); // logarithmic probabilities
    std::vector<double> probs(NL,1.);
    for (size_t l=0; l<NL; ++l)
    {
        for (size_t k=0; k<K; ++k)
        {
            int binIdx = findFeatureBinIdx(feats,k);
            probs[l] *= Ps[l].at<double>(k,binIdx)+0.1;
            logProbs[l] += std::log(Ps[l].at<double>(k,binIdx)+0.2);
        }

        //logProbs[l] = std::log(probs[l]);
        probs[l] = std::pow(2.71828,logProbs[l]);


        pq.push(std::make_pair(l,logProbs[l]));
    }

//    cvip::Image::writeVectorToFile(probs,"/home/v/Desktop/probs.dat");

    Pr tp = pq.top();

    if (probability != nullptr) {
        *probability = tp.second;
    }

    if (totProbability != nullptr) {
        *totProbability = std::accumulate(logProbs.begin(), logProbs.end(), 0.);
    }

    if (convProbability != nullptr) {
        *convProbability = logProbs[label2ix(0,0,0,0)];
    }

    return tp.first;
}

/**
 * @brief Model::estimateLabelNB
 * @param tuple - pair of images
 * @param probability - (opt) probability
 * @param convProbability - (opt) prob. that we've converged (i.e. Label = (0,0,0,0))
 * @param totProbability - sum of probabilities (for normalization
 * @return
 */
int Model::estimateLabelNB2(const cvip::Mat3& mat, double* probability, double* convProbability, double* totProbability) const
{
    int64 t1 = cv::getTickCount();
    std::vector<double> tmp = F->computeFeatures(mat);
#ifdef VERB_TIMING
    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs features" << std::endl;
#endif

/*
    for (size_t i=0; i<tmp.size(); ++i) {
        std::cout << tmp[i] << '\t';
    }

    std::cout << std::endl;
*/
    cv::Mat feats = cvip::Image::vectorToColumn(tmp);

    t1 = cv::getTickCount();
    int out =  estimateLabelNB(feats, probability, convProbability, totProbability);

#ifdef VERB_TIMING
    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs estimation" << std::endl;
#endif
    return out;
}



bool Model::isLabelOutOfRange(const Label& l) const
{
    return (fabs(l.tx) > 1.75*t ||
            fabs(l.ty) > 1.75*t ||
            fabs(l.sc) > 1.75*sc ||
            fabs(l.r) > 1.75*r);
}

void Model::increaseNumOfSamples(const Label &l)
{
    int lix = label2ix(l);
    if (samplesPerLabel[lix] < numSamplesPerLabel)
    {
        samplesPerLabel[lix]++;
        if (samplesPerLabel[lix] == numSamplesPerLabel)
            filledLabels.insert(lix);
    }
}



/**
 * @brief ModelsCascade::ModelsCascade
 * @param featuresKey
 * @param modelKeys
 */
/*
ModelsCascade::ModelsCascade(const vector<tuple<string, string> > &modelKeys)
{
    thresholdRefIdx = 0;

    for (size_t i=0; i<modelKeys.size(); ++i) {
        std::tuple<string, string> tmp = *(modelKeys.begin()+i);
        GaborBank* Gptr = new GaborBank(std::get<0>(tmp));
        models.push_back(Model(Gptr, std::get<1>(tmp)));

        //! the threshold of the cascade is the threshold of the last model w/ temp window == 2
        if (Gptr->getTimeWindow() == 2)
            thresholdRefIdx = i;
    }

    modelForThreshold = &(models[thresholdRefIdx]);
}
*/

/**
 * @brief ModelsCascade::ModelsCascade
 * @param featuresKey
 * @param modelKeys
 */
ModelsCascade::ModelsCascade(const vector<tuple<FeatureExtractor *, std::string, MultiMDN *> > &modelKeys)
{
    thresholdRefIdx = 0;

    for (size_t i=0; i<modelKeys.size(); ++i) {
        auto tmp = *(modelKeys.begin()+i);

        //! the threshold of the cascade is the threshold of the last model w/ temp window == 2
        if (std::get<0>(tmp)->getTimeWindow() == 2)
            thresholdRefIdx = i;

        models.push_back(Model(std::get<0>(tmp), std::get<1>(tmp), std::get<2>(tmp)));
    }

    modelForThreshold = &(models[thresholdRefIdx]);
}


