#include "GaborBank.hpp"
#include "Utility.hpp"
#include <cmath>
#include <chrono>

using cvip::PI;

/**
 * @brief GaborBank::GaborBank
 * @param _numPartsX
 * @param _numPartsY
 * @param _featureType
 * @param _normalisationType
 * @param _numScales
 * @param _scaleBitset - a string that will define which of the 8 filters will be usedx
 */
GaborBank::GaborBank(size_t _numPartsX,
                     size_t _numPartsY,
                     const std::string& _featureType,
                     const std::string& _normalisationType,
                     int _numScales,
                     const std::string& _scalesBitset,
                     const std::string& _orientsBitset)
    : numPartsX(_numPartsX),
      numPartsY(_numPartsY),
      timeWindow(2),
      featureType(_featureType),
      normalisationType(_normalisationType),
      numScales(_numScales),
      scalesBitset(_scalesBitset),
      orientsBitset(_orientsBitset),
      orients({0., PI/4, PI/2, 3*PI/4, PI, 5*PI/4, 3*PI/2, 7*PI/4})
{
    uniqueKey = computeUniqueKey(featureType, normalisationType, 2, numPartsX, scalesBitset.to_string(), orientsBitset.to_string());
    createFilters();
}

GaborBank::GaborBank(const std::string& featuresPath)
    : orients({0., PI/4, PI/2, 3*PI/4, PI, 5*PI/4, 3*PI/2, 7*PI/4})
{
    std::vector<std::string> l = split(featuresPath, '-');
    size_t cnt = 0;
    timeWindow = std::atoi(l[cnt++].c_str());
    numPartsX = std::atoi(l[cnt++].c_str());
    numPartsY = std::atoi(l[cnt++].c_str());
    scalesBitset =  std::bitset<5>(l[cnt++]); // probably not needed anymore
    orientsBitset =  std::bitset<8>(l[cnt++]); // probably not needed anymore
    featureType = l[cnt++];
    normalisationType = l[cnt++];

    uniqueKey = featuresPath;

    createFilters();
}

std::string GaborBank::computeUniqueKey(const string &_featureType,
                                        const string &_normalisationType,
                                        const size_t _timeWindow,
                                        const size_t _numParts,
                                        const string &_scalesBitsetStr,
                                        const string &_orientsBitsetStr)
{

    std::stringstream ss;
    //    ss << _numParts << "-" << _numParts << "-" <<  _scalesBitsetStr << "-" << _orientsBitsetStr << "-" <<rfw::imW << "-" << _featureType << "-" << _normalisationType;
    ss << _timeWindow  << "-" << _numParts << "-" << _numParts << "-" <<  _scalesBitsetStr << "-" << _orientsBitsetStr << "-" << _featureType << "-" << _normalisationType;

    return ss.str();
}


std::vector<double> GaborBank::computeFeatures(const cvip::Mat3& mat) const
{

    int64 t1 = cv::getTickCount();
    using std::vector;
    vector<double> out;

    vector<double> featuresOrig = _computeFeatures(mat);
    vector<vector<double> > featuresRef;

    vector<double> sumRefFeatures(featuresOrig.size(), 0);

    for (size_t t=0; t<mat.T(); ++t) {
        cvip::Mat3 tmp = mat.syntehsiseFromTthFrame(t);

        featuresRef.push_back(_computeFeatures(tmp));

        for (size_t i=0; i<featuresOrig.size(); ++i)
            sumRefFeatures[i] += featuresRef.back()[i];
    }

    for (size_t i=0; i<featuresOrig.size(); ++i) {
         //       out.push_back(featuresOrig[i]/((sumRefFeatures[i]+0.5)/mat.T()));
        out.push_back(featuresOrig[i]/(cvip::EPS+(sumRefFeatures[i])/mat.T()));
    }
#ifdef VERB_TIMING
    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs features" << std::endl;
#endif
    return out;
}






std::vector<cv::Mat> GaborBank::computeEnergies(tuple<Mat, Mat>& tup, const std::vector<double>& _scales,
                                                const std::vector<tuple<Mat,Mat> >& _reFilters,
                                                const std::vector<tuple<Mat,Mat> >& _imFilters)
{
    using std::get;

    std::vector<double> features;
    std::vector<cv::Mat> ens;
    auto system_start = std::chrono::system_clock::now();

    for (double scale : _scales)
    {
        cv::Mat im1 = get<0>(tup).clone();
        cv::Mat im2 = get<1>(tup).clone();

        if (scale != 1.)
        {
            cv::resize(im1,im1,cv::Size(),scale,scale,cv::INTER_LANCZOS4);
            cv::resize(im2,im2,cv::Size(),scale,scale,cv::INTER_LANCZOS4);
        }

        std::vector<std::vector<double> >  tmp(_reFilters.size(), std::vector<double>());

        //        #pragma omp parallel
        {

            int i;
            //            #pragma omp for nowait
            for (i=0; i<_reFilters.size(); ++i)
            {
                cv::Mat yre1,yre2,yim1,yim2;
                cv::filter2D(im1,yre1,im1.depth(),get<1>(_reFilters[i]));
                cv::filter2D(im2,yre2,im2.depth(),get<0>(_reFilters[i]));
                cv::Mat yre = yre1+yre2;

                cv::filter2D(im1,yim1,im1.depth(),get<1>(_imFilters[i]));
                cv::filter2D(im2,yim2,im2.depth(),get<0>(_imFilters[i]));
                cv::Mat yim = yim1+yim2;

                // compute gabor motion "energy"
                cv::Mat en;
                cv::pow(yre,2.,yre);
                cv::pow(yim,2.,yim);
                cv::pow(yre+yim,0.5,en);

                ens.push_back(en);
            }
        }
    }

    auto diff = std::chrono::system_clock::now() - system_start;
    auto sec = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
    // std::cout << "this program runs: " << sec.count() << " ms" << std::endl;

    return ens;
}

/**
 * @brief GaborBank::computeNumFeatures - compute feature vector length from given feature combination
 * @param _featureType
 * @param _numParts
 * @param orients
 * @param scales
 * @return
 */
size_t GaborBank::computeNumFeatures(const std::string& _featureType,
                                     size_t _numParts,
                                     const std::vector<double>& orients,
                                     const std::vector<double>& scales)
{
    size_t kFeatures = (_featureType == "meanstd") ? 2 : 1;
    return kFeatures * _numParts * _numParts * orients.size() * scales.size();
}


double GaborBank::signalEnergy(const std::vector<double> &x) const
{
    double energy = 0;
    double scale = 0.;
    size_t i;
    for (i=0; i<x.size(); ++i)
    {
        double tmp = x[i];
        scale = 1.+std::floor((double)i/(numPartsX*numPartsY*orients.size()));
//        energy += tmp*tmp*scale*scale;
        energy += tmp*tmp;
    }

//    energy = std::sqrt(energy);
    return energy;
}


/**
 * @brief GaborBank::_computeFeatures -- compute features straight from image pair
 * @param tup - image pair
 * @return
 */
std::vector<double> GaborBank::_computeFeatures(const cvip::Mat3& mat3d) const
{


    using std::get;

    std::vector<double> features;

    if (mat3d.frames.size() != timeWindow) {
        std::cerr << "Gabor ERROR! Time Window different from input sequence size!" << std::endl;
        exit(-1);
    }

    for (double scale : scales)
    {
        std::vector<cv::Mat> scaledIms;

        for (size_t i=0; i<mat3d.frames.size(); ++i)
            scaledIms.push_back(mat3d.frames[i].clone());

        if (scale != 1.)
        {

            for (size_t i=0; i<mat3d.frames.size(); ++i)
                cv::resize(scaledIms[i],scaledIms[i],cv::Size(),scale,scale,cv::INTER_LANCZOS4);
        }

        std::vector<std::vector<double> >  tmp(reFilters.size(), std::vector<double>());


//#pragma omp parallel
        {

            int i;
//#pragma omp for nowait
            for (i=0; i<reFilters.size(); ++i)
            {

                int T = scaledIms.size();
                cv::Mat yreT,yimT;

                cv::filter2D(scaledIms[T-1],yreT,scaledIms[T-1].depth(),reFilters2[i][0]);
                cv::filter2D(scaledIms[T-1],yimT,scaledIms[T-1].depth(),imFilters2[i][0]);



                // compute gabor motion "energy"
                 cv::Mat en(yreT.rows,yreT.cols,CV_64FC1, cv::Scalar::all(0));


                for (int t=0; t<T-1; ++t) {
                    cv::Mat _yre,_yim;
                    cv::filter2D(scaledIms[t],_yre,scaledIms[t].depth(),reFilters2[i][1]);
                    cv::filter2D(scaledIms[t],_yim,scaledIms[t].depth(),imFilters2[i][1]);

                    _yre = _yre+yreT;
                    _yim = _yim+yimT;


                    cv::Mat _en;

                    cv::pow(_yre,2.,_yre);
                    cv::pow(_yim,2.,_yim);
                    cv::pow(_yre+_yim,0.5,_en);
                    en = en+_en;
                }


                tmp[i] = pool(en, featureType, numPartsX);
            }
        }

        for (size_t i=0; i<tmp.size(); ++i)
            for (size_t k=0; k<tmp[i].size(); ++k)
            {
                if (std::isnan(tmp[i][i]))
                {
                    int asdasd=1;
                }
                features.push_back(tmp[i][k]);
            }

    }


    return features;
}






/**
 * @brief GaborBank::_computeFeatures -- compute features straight from image pair
 * @param tup - image pair
 * @return
std::vector<double> GaborBank::_computeFeatures(const cvip::Mat3& mat3d) const
{
    using std::get;

    std::vector<double> features;

    if (mat3d.frames.size() != timeWindow) {
        std::cerr << "Gabor ERROR! Time Window different from input sequence size!" << std::endl;
        exit(-1);
    }

    for (double scale : scales)
    {
        std::vector<cv::Mat> scaledIms;

        for (size_t i=0; i<mat3d.frames.size(); ++i)
            scaledIms.push_back(mat3d.frames[i].clone());

        if (scale != 1.)
        {

            for (size_t i=0; i<mat3d.frames.size(); ++i)
                cv::resize(scaledIms[i],scaledIms[i],cv::Size(),scale,scale,CV_INTER_LANCZOS4);
        }

        std::vector<std::vector<double> >  tmp(reFilters.size(), std::vector<double>());
        int T = scaledIms.size();

        std::vector<cv::Mat> yreTs, yimTs;

        for (size_t i=0; i<reFilters.size(); ++i)
        {
            yreTs.push_back(cv::Mat(0,0,CV_64FC1,cv::Scalar::all(0)));
            yimTs.push_back(cv::Mat(0,0,CV_64FC1,cv::Scalar::all(0)));
        }

        for (int t=0; t<T-1; ++t)
        {
//#pragma omp parallel
            {

                int i;
//#pragma omp for nowait
                for (i=0; i<reFilters.size(); ++i)
                {

                    cv::Mat yreT,yimT;

                    if (yreTs[i].cols)
                    {
                        yreT = yreTs[i];
                        yimT = yimTs[i];
                    }
                    else
                    {
                        cv::filter2D(scaledIms[T-1],yreT,scaledIms[T-1].depth(),reFilters2[i][0]);
                        cv::filter2D(scaledIms[T-1],yimT,scaledIms[T-1].depth(),imFilters2[i][0]);
                        yreTs[i] = yreT;
                        yimTs[i] = yimT;
                    }

                    // compute gabor motion "energy"
                    cv::Mat en(yreT.rows,yreT.cols,CV_64FC1, cv::Scalar::all(0));


                    cv::Mat _yre,_yim;
                    cv::filter2D(scaledIms[t],_yre,scaledIms[t].depth(),reFilters2[i][1]);
                    cv::filter2D(scaledIms[t],_yim,scaledIms[t].depth(),imFilters2[i][1]);

                    _yre = _yre+yreT;
                    _yim = _yim+yimT;


                    cv::pow(_yre,2.,_yre);
                    cv::pow(_yim,2.,_yim);
                    cv::pow(_yre+_yim,0.5,en);

                    tmp[i] = pool(en, featureType, numPartsX);
                }
            }

            for (size_t i=0; i<tmp.size(); ++i)
                for (size_t k=0; k<tmp[i].size(); ++k)
                    features.push_back(tmp[i][k]);

        }
    }

    return features;
}



 */















void GaborBank::extractSlice( const cv::Mat& image3d, const int z, cv::Mat& slice )
{
    // create the roi
    cv::Range ranges[3];
    ranges[2] = cv::Range::all();
    ranges[1] = cv::Range::all();
    ranges[0] = cv::Range( z, z+1);

    // get the roi from the image;
    // calling clone() makes sure the data is continuous
    slice = image3d(ranges).clone();

    // create a temporarily 2d image and copy its size
    // to make our image a real 2d image
    cv::Mat temp2d;
    temp2d.create(2, &(image3d.size[2]), image3d.type());
    slice.copySize(temp2d);
}


/**
             * @brief GaborBank::createFilter
             * @param w - filter width
             * @param h - filter height
             * @param theta - filter orientation
             * @param v - filter velocity
             * @param phi - quadrature angle (i.e. if phi==0 real, if phi==pi/2 imag filter)
             * @return
             */
std::tuple<cv::Mat, cv::Mat> GaborBank::createFilterTuple(int w, int h, double th, double v, double phi)
{
    double lam = 2*std::sqrt(1+v*v);
    double sig = lam*0.56;
    double gam = 0.75; //% 0.5
    double mut = 1.75;
    double tau = 2.75;

    double vc = 0; // the other alternative is setting vc = v -- see Petkov&Subramanian 2007
    //    double vc = v;
    // the output filter tuple
    // g = zeros(yn,xn,tn);

    std::vector<cv::Mat> tmp;
    for (size_t i=0; i<timeWindow; ++i)
        tmp.push_back(cv::Mat::zeros(h,w,CV_64FC1));

    // start from t=1, this is what the gabor formula needs
    for (int t=1; t<=timeWindow; ++t)
    {
        for (int y=std::ceil(-h/2)+1; y<floor(h/2.); y++)
        {
            for (int x=std::ceil(-w/2)+1; x<floor(w/2.); x++)
            {
                using cvip::PI;

                double xd = x*cos(th)+y*sin(th);
                double yd = -x*sin(th)+y*cos(th);

                double val = gam*exp(-(pow(xd+vc*t,2.)+pow(yd,2)*pow(gam,2))/(2*pow(sig,2)));
                val *= cos(phi+2.*PI*(xd+v*t)/lam);
                val *= exp(-(pow(t-mut,2.))/(2.*pow(tau,2.)))/(tau*sqrt(2.*PI)*2.*PI*pow(sig,2));

                tmp[t-1].at<double>(y+std::ceil(h/2),x+std::ceil(w/2)) = val;
            }
        }

        /*

                    cv::Mat ttt = cvip::Image::doubleToUchar(tmp[t-1]);
                    cv::resize(ttt,ttt,cv::Size(),5,5, CV_INTER_LANCZOS4);
                    cv::imshow("g1",ttt);
                    std::cout << t << std::endl;
                    cv::waitKey(0);
                    */
    }

    /*
    tmp[0] = tmp[0](cv::Rect(3,3,6,6));
    tmp[1] = tmp[1](cv::Rect(3,3,6,6));
*/

    return std::tuple<cv::Mat, cv::Mat>(tmp[0],tmp[1]);
}



/**
             * @brief GaborBank::createFilter
             * @param w - filter width
             * @param h - filter height
             * @param theta - filter orientation
             * @param v - filter velocity
             * @param phi - quadrature angle (i.e. if phi==0 real, if phi==pi/2 imag filter)
             * @return
             */
std::vector<cv::Mat> GaborBank::createFilterTuple(int w, int h, size_t T, double th, double v, double phi)
{
    double lam = 2*std::sqrt(1+v*v);
    double sig = lam*0.56;
    double gam = 0.75; //% 0.5
    double mut = 1.75;
    double tau = 2.75;

    double vc = 0; // the other alternative is setting vc = v -- see Petkov&Subramanian 2007
    //    double vc = v;
    // the output filter tuple
    // g = zeros(yn,xn,tn);

    std::vector<cv::Mat> tmp;
    for (size_t i=0; i<T; ++i)
        tmp.push_back(cv::Mat::zeros(h,w,CV_64FC1));

    // start from t=1, this is what the gabor formula needs
    for (int t=1; t<=T; ++t)
    {
        for (int y=std::ceil(-h/2)+1; y<floor(h/2.); y++)
        {
            for (int x=std::ceil(-w/2)+1; x<floor(w/2.); x++)
            {
                using cvip::PI;

                double xd = x*cos(th)+y*sin(th);
                double yd = -x*sin(th)+y*cos(th);

                double val = gam*exp(-(pow(xd+vc*t,2.)+pow(yd,2)*pow(gam,2))/(2*pow(sig,2)));
                val *= cos(phi+2.*PI*(xd+v*t)/lam);
                val *= exp(-(pow(t-mut,2.))/(2.*pow(tau,2.)))/(tau*sqrt(2.*PI)*2.*PI*pow(sig,2));

                tmp[t-1].at<double>(y+std::ceil(h/2),x+std::ceil(w/2)) = val;
            }
        }
        /*
                    cv::Mat ttt = cvip::Image::doubleToUchar(tmp[t-1]);
                    cv::resize(ttt,ttt,cv::Size(),5,5, CV_INTER_LANCZOS4);
                    cv::imshow("g1",ttt);
                    std::cout << t << std::endl;
                    cv::waitKey(0);
                    */
    }

    /*
    tmp[0] = tmp[0](cv::Rect(3,3,6,6));
    tmp[1] = tmp[1](cv::Rect(3,3,6,6));
    */
    return tmp;
}





void GaborBank::createFilters()
{
    std::vector<double> allScales({1.,1./2,1./4,1./6,1./8});
    std::vector<double> allOrients({0., PI/4, PI/2, 3*PI/4, PI, 5*PI/4, 3*PI/2, 7*PI/4});

    // reverse the vectors to make them intuitive -- bitset will read bits right-to-left
    std::reverse(allScales.begin(), allScales.end());
    std::reverse(allOrients.begin(), allOrients.end());

    for (int i=scalesBitset.size(); i>=0; --i) {
        if (scalesBitset[i])
            scales.push_back(allScales[i]);
    }

    orients.clear();
    for (int i=orientsBitset.size(); i>=0; --i) {
        if (orientsBitset[i])
            orients.push_back(allOrients[i]);
    }

    for (size_t i=0; i<orients.size(); ++i)
    {
        reFilters.push_back(createFilterTuple(13, 13, orients[i], 1, 0));
        imFilters.push_back(createFilterTuple(13, 13, orients[i], 1, cvip::PI/2.));

        reFilters2.push_back(createFilterTuple(13, 13, timeWindow, orients[i], 1, 0));
        imFilters2.push_back(createFilterTuple(13, 13, timeWindow, orients[i], 1, cvip::PI/2.));
    }
    return;
}
