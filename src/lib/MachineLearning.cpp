#include "MachineLearning.hpp"

/**
 * @brief MDN::init initialize the model; load weights etc.
 * @param modelPath -- path to the folder that contains model files
 */
void MDN::init(const std::string &modelPath)
{
    std::string w1path = modelPath+"/w1";
    std::string w2path = modelPath+"/w2";
    std::string b1path = modelPath+"/b1";
    std::string b2path = modelPath+"/b2";
    std::string cfgpath = modelPath+"/config";
    std::string meanpath = modelPath+"/mean";
    std::string stdpath = modelPath+"/std";

    // read basic parameters -- see immediate below
    std::vector<double> cfg = cvip::Image::readFileToVector(cfgpath);

    D = (int) cfg[0];
    Ntarget = (int) cfg[1];
    Nout = (int) cfg[2];
    Nhidden = (int) cfg[3];
    K = (int) cfg[4];

    // read matrices
    W1 = cvip::Image::readFromFile(w1path, D, Nhidden);
    W2 = cvip::Image::readFromFile(w2path, Nhidden, Nout);

    // read matrices
    b1 = cvip::Image::readFromFile(b1path, 1, Nhidden);
    b2 = cvip::Image::readFromFile(b2path, 1, Nout);



    /*
    // read matrices
    mean = cvip::Image::readFromFile(meanpath, 1, D);
    std = cvip::Image::readFromFile(stdpath, 1, D);

    std::vector<double> meanTmp = cvip::Image::columnToVector(mean);
    std::vector<double> stdTmp = cvip::Image::columnToVector(std);

*/
//    W1 = cvip::Image::readFromFile();

}


/**
 * @brief MDN::forward forward propagate an input
 * @param x -- input vector
 * @return network output
 */
std::vector<double> MDN::forward(const cv::Mat &x) const
{
    /**
    cv::Mat f = x-mean;
    for (size_t i=0; i<f.cols; ++i)
        f.at<double>(0,i) = f.at<double>(0,i)/std.at<double>(0,i);
        **/

    cv::Mat z = x*W1+b1;
    std::vector<double> zTmp = cvip::Image::columnToVector(z);
    for (size_t i=0; i<z.cols; ++i)
        z.at<double>(0,i) = std::tanh(z.at<double>(0,i));

    cv::Mat a = z*W2+b2;

    return cvip::Image::columnToVector(a);
}

/**
 * @brief MDN::predict perform regression: return the mode of the mixture component with the highest posterior
 * @param x -- input vector
 * @return
 */
std::vector<double> MDN::predict(const cv::Mat &x, double* prior) const
{
    // first propagate forward through net
    std::vector<double> y = forward(x);

    std::vector<double> priors;
    std::vector<std::vector<double> > centers;
    std::vector<double> widths;

    // constants to avoid float math traps
    double maxcut = 700.;
    double mincut = -700.;

    std::vector<double> tmp;
    double sumPrior = 0;
    for (size_t i=0; i<K; ++i)
    {
        double ttmp = y[i];
        ttmp = std::min<double>(ttmp, maxcut);
        ttmp = std::max<double>(ttmp, mincut);
        ttmp = std::exp(ttmp);
        tmp.push_back(ttmp);
        sumPrior += ttmp;
    }

    // load priors -- these will decide which is the most likely component
    for (size_t i=0; i<K; ++i)
        priors.push_back(tmp[i]/sumPrior);

    size_t idxCentresStart = K;
    size_t idxCentresEnd = idxCentresStart+K*Ntarget;

    // load centres -- the centres of for each component
    for (size_t k=0; k<K; ++k)
    {
        std::vector<double> ttmp;
        for (size_t t=0; t<Ntarget; ++t)
        {
            size_t idx = idxCentresStart+k*Ntarget+t;
            ttmp.push_back(y[idx]);
        }

        centers.push_back(ttmp);
    }

    // load widths -- covariances

    size_t idxWidthsStart = idxCentresEnd;
    size_t idxWidthEnd = y.size();

    for (size_t i=idxWidthsStart; i<idxWidthEnd; ++i)
        widths.push_back(y[i]);

    int idxMax = max_element(priors.begin(), priors.end())-priors.begin();
    if (nullptr != prior)
        *prior = priors[idxMax];

    centers[idxMax][2] = centers[idxMax][2]/100.;

    return centers[idxMax];

}















/**
 * @brief MultiMDN::findNearestIdx - generic function for finding the index of the nearest double in a vector
 * @param vec - vector to search in
 * @param val - value to match
 * @return
 */
long MultiMDN:: findNearestIdx(const std::vector<double>& vec, double val)
{
    auto _x = min_element(begin(vec), end(vec), [=] (double x, double y) {
        return fabs(x-val) < fabs(y-val);
    });

    long x = std::distance(begin(vec),_x);
    return x;
}



/**
 * @brief MDN::init initialize the model; load weights etc.
 * @param modelPath -- path to the folder that contains model files
 */
void MultiMDN::init(const std::string &modelPath)
{

    std::string cfgPath = modelPath+"/config";

    // read basic parameters -- see immediate below
    std::vector<double> cfg = cvip::Image::readFileToVector(cfgPath);

    K = (int) cfg[5];

    for (size_t k=0; k<K; ++k)
    {
        std::stringstream ss;
        ss << modelPath << '/' << std::setw(3) << std::setfill('0') << (k+1);

        std::string netPath = ss.str();
        nets.push_back(MDN(netPath));

        std::string centrePath = netPath+"/centre";
        centres.push_back(cvip::Image::readFileToVector(centrePath).back());
        //!std::cout << centres.back() << std::endl;
    }
}



/**
 * @brief MDN::predict perform regression: return the mode of the mixture component with the highest posterior
 * @param x -- input vector
 * @return
 */
std::vector<double> MultiMDN::predict(const cv::Mat &x, double energy, double* prior, int* clusterIdxOutput) const
{
    int clusterIdx = findNearestIdx(centres, energy);

    if (nullptr != clusterIdxOutput)
        *(clusterIdxOutput) = clusterIdx;

    return nets[clusterIdx].predict(x, prior);
}



























#include "MachineLearning.hpp"

/**
 * @brief MDN::init initialize the model; load weights etc.
 * @param modelPath -- path to the folder that contains model files
 */
void MLP::init(const std::string &modelPath)
{
    std::string w1path = modelPath+"/w1";
    std::string w2path = modelPath+"/w2";
    std::string b1path = modelPath+"/b1";
    std::string b2path = modelPath+"/b2";
    std::string cfgpath = modelPath+"/config";
    std::string ftspath = modelPath+"/fts";
    std::string meanpath = modelPath+"/mean";
    std::string stdpath = modelPath+"/std";
    std::string thresholdpath = modelPath+"/threshold";


    // read basic parameters -- see immediate below
    std::vector<double> cfg = cvip::Image::readFileToVector(cfgpath);

    D = (int) cfg[0];
    Ntarget = (int) cfg[1];
    Nout = (int) cfg[2];
    Nhidden = (int) cfg[3];

    std::vector<double> tmp = cvip::Image::readFileToVector(ftspath);

    if (tmp.size())
    {
    for (size_t i=0; i<tmp.size(); ++i)
        selectedFeats.push_back(tmp[i]);
    }
    else
    {
        for (size_t i=0; i<D; ++i)
            selectedFeats.push_back(i);
    }

    D = selectedFeats.size();

    // read matrices
    W1 = cvip::Image::readFromFile(w1path, D, Nhidden);
    W2 = cvip::Image::readFromFile(w2path, Nhidden, Nout);

    // read matrices
    b1 = cvip::Image::readFromFile(b1path, 1, Nhidden);
    b2 = cvip::Image::readFromFile(b2path, 1, Nout);


    if (cvip::Image::exists(thresholdpath))
    {
        cv::Mat tmp = cvip::Image::readFromFile(thresholdpath, 1, 1);
        threshold = tmp.at<double>(0,0);
    }

    /*
    // read matrices
    mean = cvip::Image::readFromFile(meanpath, 1, D);
    std = cvip::Image::readFromFile(stdpath, 1, D);

    std::vector<double> meanTmp = cvip::Image::columnToVector(mean);
    std::vector<double> stdTmp = cvip::Image::columnToVector(std);

*/
//    W1 = cvip::Image::readFromFile();

}


/**
 * @brief MDN::forward forward propagate an input
 * @param x -- input vector
 * @return network output
 */
std::vector<double> MLP::forward(const cv::Mat &_x) const
{
    /**
    cv::Mat f = x-mean;
    for (size_t i=0; i<f.cols; ++i)
        f.at<double>(0,i) = f.at<double>(0,i)/std.at<double>(0,i);
        **/

    cv::Mat x(1,selectedFeats.size(), CV_64FC1);

    for (size_t i=0; i<selectedFeats.size(); ++i)
        x.at<double>(0, i) = _x.at<double>(0, selectedFeats[i]);

    std::vector<double> xTmp = cvip::Image::columnToVector(x);

    cv::Mat z = x*W1+b1;
    std::vector<double> zTmp = cvip::Image::columnToVector(z);
    for (size_t i=0; i<z.cols; ++i)
        z.at<double>(0,i) = std::tanh(z.at<double>(0,i));

    cv::Mat a = z*W2+b2;

//    std::cout << z.at<double>(0,0) << '\t' << a.at<double>(0,0) << std::endl;

    return cvip::Image::columnToVector(a);
}

/**
 * @brief MDN::predict perform regression: return the mode of the mixture component with the highest posterior
 * @param x -- input vector
 * @return
 */
std::vector<double> MLP::predict(const cv::Mat &x, double* prior) const
{
    // first propagate forward through net
    std::vector<double> y = forward(x);

    if (y.size() > 1)
    {
        y[2] = y[2]/100.;
    }

    return y;
}

/**
 * @brief MLP::sigmoid
 * @param x
 * @return
 */
double MLP::sigmoid(double x) const
{
    double maxcut = -log(cvip::EPS);
    double mincut = -log(1/cvip::EPS -1);

    x = std::min(x, maxcut);
    x = std::max(x, mincut);

    return  1./(1+exp(-x));
}





























































































