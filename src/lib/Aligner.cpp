#include "Models.hpp"
#include "Aligner.hpp"
#include <algorithm>
#include <chrono>
#include "OpticalFlow.hpp"
#include "LightGaborBank.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/videoio.hpp>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core_c.h>

#include <stdio.h>
//#define VERB 1
//#define EXTRAVERB 1  // Verbose output -- see registration output etc/


using std::vector;
using std::tuple;
using cv::Mat;



/**
 * @brief convert2doubleColumn -- OpenFace gives us a single long column vector (DOUBLE) where X and Y
 * coordinates are stacked. We convert it to a 2-column FLOAT vector (1st col: X, 2nd col: Y).
 *
 * @param X
 * @return
 */
cv::Mat convert2doubleColumn(const cv::Mat& X)
{
    int nFeats = X.rows/2;
    cv::Mat Y(nFeats, 2, CV_32FC1, cv::Scalar::all(0));

    for (size_t i=0; i<nFeats; ++i)
    {
        Y.at<float>(i, 0) = X.at<double>(i, 0);
        Y.at<float>(i, 1) = X.at<double>(i+nFeats,0);
    }

    return Y;
}









// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;








#ifdef GUI





// Drawing landmarks on a face image
void DrawLandmarks(cv::Mat img, const cv::Mat& shape2D)
{
    const int draw_multiplier = 1 << 4;
    const int draw_shiftbits = 4;

    int n;

    if(shape2D.cols == 2)
        n = shape2D.rows;
    else if(shape2D.cols == 1)
        n = shape2D.rows/2;


    for( int i = 0; i < n; ++i)
    {

        if (i != 36 && i != 39 && i != 42 && i != 45)
            continue;

        cv::Point featurePoint;
        if(shape2D.cols == 1)
        {
            featurePoint = cv::Point(cvRound(shape2D.at<double>(i) * (double)draw_multiplier), cvRound(shape2D.at<double>(i + n) * (double)draw_multiplier));
        }
        else
        {
            featurePoint = cv::Point(cvRound(shape2D.at<float>(i, 0) * (double)draw_multiplier), cvRound(shape2D.at<float>(i, 1) * (double)draw_multiplier));
        }

        // A rough heuristic for drawn point size
        int thickness = (int)std::ceil(5.0* ((double)img.cols) / 640.0);
        int thickness_2 = (int)std::ceil(1.5* ((double)img.cols) / 640.0);

        cv::circle(img, featurePoint, 1 * draw_multiplier, cv::Scalar(0, 0, 255), thickness, CV_AA, draw_shiftbits);
        cv::circle(img, featurePoint, 1 * draw_multiplier, cv::Scalar(255, 0, 0), thickness_2, CV_AA, draw_shiftbits);
    }
}

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, cv::Mat_<float>& depth_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

    // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
    double detection_certainty = face_model.detection_certainty;
    bool detection_success = face_model.detection_success;

    double visualisation_boundary = 1.2;

    // Only draw if the reliability is reasonable, the value is slightly ad-hoc
    if (detection_certainty < visualisation_boundary)
    {
        cv::Mat landmarks = convert2doubleColumn(face_model.detected_landmarks);

        LandmarkDetector::Draw(captured_image, face_model);

        double vis_certainty = detection_certainty;
        if (vis_certainty > 1)
            vis_certainty = 1;
        if (vis_certainty < -1)
            vis_certainty = -1;

        vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

        // A rough heuristic for box around the face width
        int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

        cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);

        // Draw it in reddish if uncertain, blueish if certain
        LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);

    }

    // Work out the framerate
    if (frame_count % 10 == 0)
    {
        double t1 = cv::getTickCount();
        fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
        t0 = t1;
    }

    // Write out the framerate on the image before displaying it
    char fpsC[255];
    std::sprintf(fpsC, "%d", (int)fps_tracker);
    string fpsSt("FPS:");
    fpsSt += fpsC;
    cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));

    if (!det_parameters.quiet_mode)
    {
        cv::namedWindow("tracking_result", 1);
        cv::imshow("tracking_result", captured_image);

        if (!depth_image.empty())
        {
            // Division needed for visualisation purposes
            imshow("depth", depth_image / 2000.0);
        }

    }
}
#endif















Aligner::Aligner(ModelsCascade* _MC, double _operationScale)
    : MC(_MC), thresholdMlp(nullptr), thresholdF(nullptr), Tw(4), deletePointers(false), operationScale(_operationScale), numIterations(-1), performedIterations(-1)
{
    deletePointers = false;
    init(_MC);
}


void Aligner::init(ModelsCascade* _MC)
{
    MC = _MC;

    // the reference timeWindow is the largest time window among all models
    timeWindow = 0;
    for (size_t i=0; i<MC->models.size(); ++i) {
        if (MC->models[i].F->getTimeWindow() > timeWindow)
            timeWindow = MC->models[i].F->getTimeWindow();
    }
}


/**
 * @brief Aligner::Aligner
 *          Hassle-free constructor for some pre-define features
 * @param patchType
 */
Aligner::Aligner(const string &patchType)
{

    thresholdF = nullptr;
    thresholdMlp = nullptr;


    deletePointers = true;

    std::string key = patchType+"-8-0.31-2-1-0.01-0.005-2-1-5";


    /*
    Fptrs.push_back(new OpticalFlow("optical_flow-5-15")); //               [0] OF
    Fptrs.push_back(new OpticalFlow("optical_flow-20-15")); //              [1] OF_face
    Fptrs.push_back(new GaborBank("2-3-3-10000-11111111-std-double")); //   [2] G2std
    Fptrs.push_back(new GaborBank("2-3-3-11100-11111111-std-double")); //   [3] G2std22
    Fptrs.push_back(new GaborBank("3-4-4-10000-11111111-std-double")); //   [4] G344
    MDNptrs.push_back(new MultiMDN("models/mdnmodels/model_1208"));           // [0] mdn
    MDNptrs.push_back(new MultiMDN("models/mdnmodels/model_OF_"+patchType));  // [1] mdnOF
    MDNptrs.push_back(new MultiMDN("models/mdnmodels/model_30-gab11100-face-owanpersie_makesingleNEW"));
    // [2] mdnGab22
    MDNptrs.push_back(new MultiMDN("models/mdnmodels/model_1-gab11100-face-onanpersie"));
    // [3] mdnG1m5
    MDNptrs.push_back(new MultiMDN("models/mdnmodels/model_9--"));            // [4] mdnGab
    MDNptrs.push_back(new MultiMDN("models/mdnmodels/model_G34_"+patchType)); // [5] mdnG32
*/

    Fptrs.push_back(new GaborBank("2-3-3-11100-11111111-std-double"));
    //    Fptrs.push_back(new LightGaborBank("2-3-3-11100-11111111-std-double"));
    MDNptrs.push_back(new MultiMDN("models/mdnmodels/model_0-gab11111-face-owanpersie_noise_strategies_add_noise_1.50"));


    std::vector<std::tuple<FeatureExtractor*, std::string, MultiMDN*> > mps;

    if (patchType == "face")
    {
        mps.push_back(make_tuple(Fptrs[0], key, MDNptrs[0]));
        "";
        /*
        mps.push_back(make_tuple(Fptrs[1], key, MDNptrs[0]) );
        mps.push_back(make_tuple(Fptrs[3], key, MDNptrs[2]) );
        mps.push_back(make_tuple(Fptrs[2], key, MDNptrs[4]) );
        mps.push_back(make_tuple(Fptrs[2], key, MDNptrs[3]) );
        mps.push_back(make_tuple(Fptrs[4], key, MDNptrs[5]) );
        */
    }
    else
    {
        mps.push_back(make_tuple(Fptrs[0],key,MDNptrs[1]));
        mps.push_back(make_tuple(Fptrs[4],key,MDNptrs[5]));
    }


    MC = new ModelsCascade(mps);

    init(MC);
}

Aligner::~Aligner()
{
    if (deletePointers)
    {
        for (size_t i=0; i<Fptrs.size(); ++i)
            delete Fptrs[i];

        for (size_t i=0; i<MDNptrs.size(); ++i)
            delete MDNptrs[i];

        delete MC;
    }
}


std::pair<bool, Label> Aligner::attemptCorrection(int twrong, std::vector<cv::Mat>& magicIms,
                                                  std::vector<Label>* estLabels, std::vector<Mat>* estWarps,
                                                  std::vector<bool>* successFlag)
{
    bool isFixed = false;
    Label l(0,0,1,0);

    int T = successFlag->size();

    for (int tp=std::max<int>(0,twrong-Tw); tp<std::min<int>(T, twrong+Tw); ++tp)
    {
        std::vector<cv::Mat> segment(1,magicIms[tp]);
        std::vector<cv::Mat> segmentCrp(1, imutil::cropCore(magicIms[tp], rfw::imWs[MC->getPatchType()]));

        if (!successFlag->at(tp))
            continue;

        segment.push_back(magicIms[twrong]);
        segmentCrp.push_back(imutil::cropCore(magicIms[twrong], rfw::imWs[MC->getPatchType()]));

        double convProb = -1;
        for (size_t m=0; m<MC->models.size(); m++)
        {
            // the registered version of the last image in segment
            cv::Mat tmp  = alignTmp(segment, m, &convProb, &l, 0, true);
        }

        if (convProb > 0.3*thresholdMlp->threshold)
        {
            //!std::cout << "#" << tp << " is a good boy!" << std::endl;
            isFixed = true;
            successFlag->at(twrong) = true;
            //            imutil::applyLabel(magicIms[twrong], l.invert());
            estLabels->at(twrong) = l;

            //            cv::Mat H1 = l.invert().getWarpMatrix(magicIms[twrong]);
            cv::Mat H1 = l.getWarpMatrix(magicIms[twrong]);
            cv::Mat rrow = (cv::Mat_<double>(1, 3) << 0,0,1);
            H1.push_back(rrow);

            cv::Mat H2 = estLabels->at(tp).getWarpMatrix(magicIms[tp]);
            H2.push_back(rrow);

            cv::Mat H = H1;
            //            H = H.inv();

            if (nullptr != estWarps)
            {
                cv::Mat W = H(cv::Rect(0,0,3,2));
                estWarps->at(twrong) = W;
            }

            //!cvip::Image::printMat(H1);
            //!cvip::Image::printMat(H);
            //            imutil::applyLabel(magicIms[twrong], estWarps->at(twrong));
            imutil::applyLabel(magicIms[twrong], l.invert().getWarpMatrix(magicIms[twrong]));
            //            imutil::applyLabel(magicIms[twrong], estWarps->at(tp));
            //            imutil::applyLabel(magicIms[twrong], l.invert());

            cv::Mat tmp1 = magicIms[tp].clone();
            cv::Mat tmp2 = magicIms[twrong].clone();

            tmp1 = imutil::cropCore(tmp1);
            tmp2 = imutil::cropCore(tmp2);

            //                cv::resize(tmp1, tmp1, cv::Size(), 3, 3);
            //                cv::resize(tmp2, tmp2, cv::Size(), 3, 3);
            /*
            for (size_t k=0; k<60; ++k)
            {
                cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",tmp1/255);
                cv::waitKey(50);
                cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",tmp2/255);
                cv::waitKey(50);
            }

*/
            break;
        } else {
            //!std::cout << "SEARCH GOES ON" << std::endl;
        }
    }

    return std::make_pair(isFixed, l);
}


std::vector<cv::Mat> Aligner::eliminateFalsePositives(const std::vector<cv::Mat>& frames, const std::vector<int>& clusterIdx)
{
    //! first find the dominant cluster idx
    std::multiset<int> ms;
    std::set<int> uniqueClusterIdx;
    for (size_t i=0; i<clusterIdx.size(); ++i)
    {
        uniqueClusterIdx.insert(clusterIdx[i]);
        ms.insert(clusterIdx[i]);
    }

    int maxIdx = -1;
    int maxIdxCnt = -1;
    for (auto it=uniqueClusterIdx.begin(); it != uniqueClusterIdx.end(); ++it)
    {
        int cnt = ms.count(*it);
        if (cnt>maxIdxCnt) {
            maxIdxCnt = ms.count(*it);
            maxIdx = *it;
        }
    }
    //! found it, it's maxIdx.

    std::vector<cv::Mat> out;

    for (size_t i=0; i<frames.size(); ++i)
    {
        if (clusterIdx[i] != maxIdx)
            continue;

        out.push_back(frames[i].clone());
    }

    return out;
}





/**
 * @brief Aligner::findReferenceToStichSegments
 * @param seg1 - seg1 must be on the left of seg2 (in terms of time, i.e. should come before seg2)
 * @param seg2
 * @return
 */
std::pair<Label, bool> Aligner::findReferenceToStichSegments(const std::vector<cv::Mat>& seg1,
                                                             const std::vector<cv::Mat> &seg2,
                                                             std::set<pair<int, int> >* pairsCheckedPtr,
                                                             int offsetSeg1, int offsetSeg2,
                                                             int* idxSeg1, int* idxSeg2)
{
    using std::map;
    using std::multimap;
    using std::pair;

    multimap<int, pair<int,int> > keys;
    int N1 = seg1.size();
    int N2 = seg2.size();

    // we'll try to stich the two segments by finding a pair<imFromSeg1,imFromSeg2> that is "registerable"
    // we'll start the search by the images that are temporally closer, so we obtain a map of keys ordered by temporal distance.
    for (int i=0; i<seg1.size(); ++i)
    {
        for (int j=0; j<seg2.size(); ++j)
        {
            int dist = fabs(i-(N1+j));

            // skip this if it was checked in the past
            if (nullptr != pairsCheckedPtr)
            {
                if (pairsCheckedPtr->end() != pairsCheckedPtr->find(pair<int,int>(i+offsetSeg1,j+offsetSeg2)))
                    continue;
            }

            if (dist > 10)
                continue;

            keys.insert(pair<int,pair<int, int> >(dist, pair<int,int>(i,j)));
        }
    }

    Label outLabel(0,0,0,0);
    bool found = false;

    for (auto it = keys.begin(); it != keys.end(); ++it)
    {
        pair<int,int> ims = it->second;

        cv::Mat im1 = seg1[std::get<0>(ims)];
        cv::Mat im2 = seg2[std::get<1>(ims)];
        auto p = std::make_tuple(im1,im2);

        double convLogLikelihood;

        std::pair<Label, bool> result = alignPair(p, &convLogLikelihood);

        if (result.second) {
            outLabel = Label(result.first);
            found = true;

            if (nullptr != idxSeg1)
                *idxSeg1 = std::get<0>(ims);

            if (nullptr != idxSeg2)
                *idxSeg2 = std::get<1>(ims);

            break;
        }
        else
        {
            if (nullptr != pairsCheckedPtr) {
                pairsCheckedPtr->insert(std::pair<int,int>(ims.first+offsetSeg1, ims.second+offsetSeg2));
            }
        }
    }

    return std::make_pair(outLabel, found);
}

std::vector<std::vector<cv::Mat> > Aligner::stichNeighbouringSegments(std::vector<std::vector<cv::Mat> >& segments, std::vector<Label>& labels, std::set<std::pair<int,int> >* checkedPairs)
{
    auto system_start = std::chrono::system_clock::now();

    int minWidth = 1000000;
    minWidth = 232;

    std::cout << "Stiching neighbouring segments. #Segments before stitching: " << segments.size() << std::endl;
    size_t segSizeBefore;

    do {
        segSizeBefore = segments.size();

        for (size_t i=0; i<segments.size()-1; ++i)
        {
            std::vector<cv::Mat>& curSeg = segments[i];
            std::vector<cv::Mat>& nextSeg = segments[i+1];

            if (curSeg.size() == 1 || nextSeg.size() == 1)
                continue;

            int offsetCurIdx = getSegmentOffset(segments,i);
            int offsetNextIdx = getSegmentOffset(segments,i+1);

            int idxCurSeg, idxNextSeg;
            auto result = findReferenceToStichSegments(curSeg, nextSeg, checkedPairs, getSegmentOffset(segments,i), getSegmentOffset(segments,i+1), &idxCurSeg, &idxNextSeg);

            // continue if couldn't stich these segments
            if (!result.second)
                continue;

            Label estLabel(result.first);

            int refWidth = nextSeg[idxNextSeg].cols;

            // otherwise we found a reference image, let's apply to the second sequence
            for (size_t j=0; j<nextSeg.size(); ++j)
            {
                nextSeg[j] = imutil::cropCore(nextSeg[j], refWidth);
                imutil::applyLabel(nextSeg[j], estLabel);
                /********
                nextSeg[j] = imutil::shift(nextSeg[j], -estLabel.tx, -estLabel.ty);
                nextSeg[j] = imutil::resizeUpscaled(nextSeg[j], 1./estLabel.sc);
                nextSeg[j] = imutil::rotate(nextSeg[j], -estLabel.r);
                *********/

                labels[offsetNextIdx+j] = labels[offsetNextIdx+j].combine(estLabel);
            }

            std::copy(nextSeg.begin(), nextSeg.end(), std::back_inserter(curSeg));

            segments.erase(segments.begin()+i+1);

            std::vector<cv::Mat> diffsBefore, diffsAfter;

            for (size_t k=0; k<curSeg.size(); ++k)
            {
                std::stringstream ss;

                //                ss << "tmp/" << std::setw(4) << std::setfill('0') << k << ".png";
                //                cv::imwrite(ss.str(),imutil::cropCore(cvip::Image::doubleToUchar(curSeg[k])));
            }

            break;
        }

        std::cout << "... number of segments is now " << segments.size() << std::endl;
    } while (segSizeBefore != segments.size() && segments.size() != 1);

    auto diff = std::chrono::system_clock::now() - system_start;
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(diff);
    std::cout << "Stiching neighbors completed. Total number of segments: " << segments.size() << " ( " << sec.count() << " seconds)" <<  std::endl;

    return segments;
}


/**
 * @brief Aligner::stichDistantSegments
 * @param segments
 * @return
 */
std::pair<std::vector<cv::Mat>, std::vector<int> > Aligner::stichDistantSegments(std::vector<std::vector<cv::Mat> >& segments, std::vector<Label>& labels, std::set<std::pair<int,int> >* checkedPairs)
{
    using std::map;
    using std::multimap;
    using std::pair;

    auto system_start = std::chrono::system_clock::now();

    // a lambda function to count the number of unique keys in a multimap, will be used later...
    auto numUniqueKeys = [] (multimap<int,pair<int,int> >& keys) {
        size_t numKeys = 0;
        for(  multimap<int,pair<int,int> >::iterator it = keys.begin(), lastkey = keys.end();
              it != lastkey; it = keys.upper_bound(it->first))
        {
            numKeys++;
        }
        return numKeys;
    };


    multimap<int, pair<int,int> > keys;


    // we'll try to stich the two segments by finding a pair<imFromSeg1,imFromSeg2> that is "registerable"
    // we'll start the search by the images that are temporally closer, so we obtain a map of keys ordered by temporal distance.
    for (int i=0; i<segments.size(); ++i)
    {
        for (int j=i; j<segments.size(); ++j)
        {
            int dist = fabs(i-j);

            keys.insert(pair<int,pair<int, int> >(dist, pair<int,int>(i,j)));
        }
    }

    std::cout << "Stiching distant segments. #Segments before stitching: " << segments.size() << std::endl;

    // initially, assign a unique key to each segment, we'll later try to merge
    std::vector<int> keysOfSegments;
    for (int i=0; i<segments.size(); ++i)
        keysOfSegments.push_back(i);

    const int NUM_ALLOWED_IT = 20;
    int IT = 0;

    bool anySegmentChanged;
    do {
        anySegmentChanged = false;


        if (IT++ > NUM_ALLOWED_IT)
            break;


        bool allKeysAreSame = true;
        int prevKey = keysOfSegments[0];
        for (size_t i=1; i<keysOfSegments.size(); ++i) {
            if (prevKey != keysOfSegments[i]) {
                allKeysAreSame = false;
                break;
            }
        }

        if (allKeysAreSame) {
            int asd =12;
            break;
        }

        for (auto it = keys.begin(); it != keys.end(); ++it)
        {
            std::vector<cv::Mat>& curSeg = segments[it->second.first];
            std::vector<cv::Mat>& nextSeg = segments[it->second.second];

            if (curSeg.size() == 1 || nextSeg.size() == 1)
                continue;

            if (keysOfSegments[it->second.first] == keysOfSegments[it->second.second])
                continue;

            int idxCurSeg,idxNextSeg;
            auto result = findReferenceToStichSegments(curSeg, nextSeg, checkedPairs, getSegmentOffset(segments, it->second.first), getSegmentOffset(segments, it->second.second), &idxCurSeg, &idxNextSeg);

            int refWidth = nextSeg[idxNextSeg].cols;

            // continue if couldn't stich these segments
            if (!result.second)
                continue;

            // we found a match, something will be updated now
            anySegmentChanged = true;

            Label estLabel(result.first);

            // now convert all the segments with the key "it->second.second"
            for (size_t i=0; i<keysOfSegments.size(); ++i)
            {
                if (keysOfSegments[i] != keysOfSegments[it->second.second])
                    continue;

                int abc = 10;
                if (i == it->second.first)
                    continue;

                std::vector<cv::Mat>& seg = segments[i];

                int segOffset = getSegmentOffset(segments, i);

                // otherwise we found a reference image, let's apply to the second sequence
                for (size_t j=0; j<seg.size(); ++j)
                {
                    seg[j] = imutil::cropCore(seg[j], refWidth);
                    imutil::applyLabel(seg[j], estLabel);

                    labels[segOffset+j] = labels[segOffset+j].combine(estLabel);
                }


                //                std::cout << "trying to combine segment << " << i << " with segment " << j << std::endl;
                keysOfSegments[i] = keysOfSegments[it->second.first];
            }

            break;
        }

        std::set<int> tmp(keysOfSegments.begin(), keysOfSegments.end());


        std::cout << "... number of unique-keyed segments: " << tmp.size() << std::endl;
    } while (anySegmentChanged);

    auto diff = std::chrono::system_clock::now() - system_start;
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(diff);

    std::set<int> tmp(keysOfSegments.begin(), keysOfSegments.end());

    std::cout << "Stiching distant segments completed. Total number of segments: " << tmp.size()  << " ( " << sec.count() << " seconds)" <<  std::endl;



    std::vector<cv::Mat> outims;
    std::vector<int> outkeys;

    for (size_t i=0; i<segments.size(); ++i)
    {
        for (size_t j=0; j<segments[i].size(); ++j)
        {
            outims.push_back(segments[i][j]);
            outkeys.push_back(keysOfSegments[i]);
        }
    }

    return make_pair(outims,outkeys);

}

std::vector<cv::Mat> Aligner::alignSequence(const std::vector<cv::Mat>& magicIms, std::vector<cv::Mat>& outMagicIms,
                                            std::vector<Label> *estLabels, std::vector<int>& registrationStatus,
                                            std::vector<cv::Mat> *diffsBefore, std::vector<cv::Mat> *diffsAfter, std::vector<cv::Mat> *estWarps, std::vector<bool> *successFlag,
                                            std::vector<Mat> *allEstWarps)
{
    std::vector<cv::Mat> out;

    bool deleteEstLabelsPtr = false;

    if (estLabels == nullptr)
    {
        deleteEstLabelsPtr = true;
        estLabels = new std::vector<Label>();
    }




    if (nullptr != allEstWarps)
    {
        Label dummyLabel(0,0,1,0);
        cv::Mat H = dummyLabel.getWarpMatrix(magicIms[0]);
        allEstWarps->push_back(H);
    }


    if (magicIms.size() <2)
        return out;

    auto tpl = alignSequenceWithCorrection(magicIms, diffsBefore, diffsAfter, nullptr, estLabels, nullptr, estWarps, successFlag, allEstWarps);

    std::vector<cv::Mat> alignedFrames = std::get<0>(tpl);
    registrationStatus = std::get<1>(tpl);

    for (size_t i=0; i<magicIms.size(); ++i)
    {
        cv::Mat tmp = magicIms[i].clone();
        cv::Mat H = estWarps->at(i);

        cv::Mat rrow = (cv::Mat_<double>(1,3) << 0,0,1);

        H.push_back(rrow);
        H = H.inv();
        H = H(cv::Rect(0,0,3,2));

        imutil::applyLabel(tmp, estLabels->at(i).invert());



        //        imutil::applyLabel(tmp, H);

        outMagicIms.push_back(tmp.clone());

        out.push_back(imutil::cropCore(tmp));
    }



    if (deleteEstLabelsPtr)
        delete estLabels;

    if (nullptr != estWarps)
    {
        estWarps->clear();
        for (size_t i=0; i<estLabels->size(); ++i)
        {
            //            cvip::Image::printMat((estWarps->at(i)));

            cv::Mat H = estLabels->at(i).invert().getWarpMatrix(magicIms[i]);
            estWarps->push_back(H);
            //!cvip::Image::printMat(H);
        }
    }


    return alignedFrames;

    return out;
}




std::vector<cv::Mat> Aligner::getRegisteredSequence(const std::vector<cv::Mat>& magicIms, std::vector<Label> &estLabels)
{
    std::vector<cv::Mat> out;

    for (size_t i=0; i<magicIms.size(); ++i)
    {
        cv::Mat tmp = magicIms[i];
        imutil::applyLabel(tmp, estLabels[i]);
        out.push_back(imutil::cropCore(tmp));
    }

    return out;
}





std::vector<cv::Mat> Aligner::readSeqClipFromVideo(const std::string& videopath, int tBegin, int tEnd, int frameSize)
{
    cv::VideoCapture cap(videopath);

    //!std::cout << videopath << std::endl;
    std::vector<cv::Mat> frames;

    int t=-1;
    while (true)
    {
        t++;

        cv::Mat frame;
        cap >> frame;

        if (!frame.cols)
            break;

        if (frames.size() == tEnd-tBegin)
            break;

        if (t < tBegin || t >= tEnd)
            continue;


        cv::resize(frame, frame, cv::Size(frameSize, frameSize));

        if (frame.channels() > 1)
            cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
        /*
        */

        frames.push_back(frame);
    }


    return frames;
}


/**
 * @brief Aligner::align  align a sequence
 * @param magicIms
 * @param beforeDiffs - (opt.) vector that contains the difference image of each pair before registration
 * @param afterDiffs - (opt.) vector that contains the difference image of each registered pair
 * @return
 */
std::vector<cv::Mat> Aligner::align(const std::vector<cv::Mat>& _magicIms, size_t ref, std::vector<cv::Mat> *beforeDiffs,
                                    std::vector<cv::Mat> *afterDiffs, std::vector<Label>* gtLabels, std::vector<Label>* estLabels,
                                    std::vector<cv::Mat> *Hs, std::vector<double> *convLogLikelihoods)
{
    std::vector<cv::Mat> magicIms(_magicIms);
    // first convert all images to double
    for (size_t i=0; i<magicIms.size(); ++i)
    {
        if (magicIms[i].type() != CV_64FC1)
            magicIms[i].convertTo(magicIms[i], CV_64FC1);
    }



    // The first label is the reference
    if (nullptr != estLabels)
    {
        estLabels->push_back(Label(0,0,1,0));
        Hs->push_back(cv::Mat::eye(2,3,CV_64FC1));
    }
    std::vector<cv::Mat> out;
    out.push_back(magicIms[0]);

    // register tuples of images
    for (int i=1; i<magicIms.size(); ++i)
    {
        auto system_start = std::chrono::system_clock::now();
        //!std::cout << "Registering pair #" << i << " out of " << magicIms.size()-1 << "..." ;

        std::vector<cv::Mat> segment, segmentCrp;

        int idxStart = std::max<int>(0, i-timeWindow+1);
        for (int j=idxStart; j<=i; ++j) {
            segment.push_back(magicIms[j]);
            segmentCrp.push_back(imutil::cropCore(magicIms[j], rfw::imWs[MC->getPatchType()]));
        }


        cv::Mat before1 = segmentCrp[segmentCrp.size()-2];
        cv::Mat before2 = segmentCrp[segmentCrp.size()-1];

        cv::Mat diffBef = before2-before1;

        if (nullptr != beforeDiffs) {
            beforeDiffs->push_back(diffBef);
        }

        // cumulative estimation, will be accumulated over cascaded models
        Label estLabel(0,0,1,0);


        for (size_t m=0; m<MC->models.size(); m++)
        {
            double convLogLikelihood;

            bool greenLight = (m == (MC->models.size()-1));
            greenLight = false;
            greenLight = true;

            // the registered version of the last image in segment
            cv::Mat tmp  = alignTmp(segment, m, &convLogLikelihood, &estLabel, 0, greenLight);
        }

        cv::Mat imRegLast = magicIms.back().clone();


        //        imutil::applyLabel(imRegLast, estLabel);
        cv::Mat H = imutil::applyLabel(imRegLast, estLabel.invert());

        std::vector<cv::Mat> controlSegment(segment);
        controlSegment[controlSegment.size()-1] = imRegLast;

        //        double refLogLikelihood = computeConvergenceLikelihood(controlSegment, MC->modelForThreshold);
        double refLogLikelihood = 50;

        //        if (refLogLikelihood < MC->getThreshold()) {
        if (refLogLikelihood < 0) {
            std::cout << "DIDN'T PASS SECURITY CHECK!!!" << std::endl;
            std::cout << "DIDN'T PASS SECURITY CHECK!!!" << std::endl;
            std::cout << "DIDN'T PASS SECURITY CHECK!!!" << std::endl;
        }

        if (nullptr != convLogLikelihoods)
            convLogLikelihoods->push_back(refLogLikelihood);

        if (nullptr != estLabels)
            estLabels->push_back(estLabel);

        if (nullptr != Hs)
            Hs->push_back(H);


        int pause(0);
        // ground truth (if available)
        if (nullptr != gtLabels && nullptr != estLabels)
        {
        }


        cv::Mat cim1After = imutil::cropCore(segment[segment.size()-2].clone(), rfw::imWs[MC->getPatchType()]);
        cv::Mat cim2After = imutil::cropCore(imRegLast.clone(), rfw::imWs[MC->getPatchType()]);

        cv::Mat diffAfter = cim2After-cim1After;


#ifdef VERB
#ifdef GUI
        std::cout << "REGISTERED PAIR" << std::endl;
        std::cout << "REGISTERED PAIR" << std::endl;
        std::cout << "REGISTERED PAIR" << std::endl;
        for (size_t k=0; k<20; ++k) {
            cv::imshow("Before: \hat{I}_t, I_{t+1}",before1/255);
            cv::waitKey(50);
            cv::imshow("Before: \hat{I}_t, I_{t+1}",before2/255);
            cv::waitKey(50);
        }
        cv::imshow("Difference BEFORE", cvip::Image::doubleToUchar(diffBef));

        for (size_t k=0; k<20; ++k) {
            cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim1After/255);
            cv::waitKey(50);
            cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim2After/255);
            cv::waitKey(50);
        }
        cv::imshow("Difference AFTER ", cvip::Image::doubleToUchar(diffAfter));
        /****

****/
#endif
#endif

        if (nullptr != afterDiffs)
            afterDiffs->push_back(diffAfter);

        magicIms[i] = imRegLast;

        out.push_back(magicIms[i]);

        auto diff = std::chrono::system_clock::now() - system_start;
        auto sec = std::chrono::duration_cast<std::chrono::seconds>(diff);

        std::cout << sec.count() << " seconds." << std::endl;
    }

    return out;
}



/**
 * @brief Aligner::align  align a sequence
 * @param magicIms
 * @param beforeDiffs - (opt.) vector that contains the difference image of each pair before registration
 * @param afterDiffs - (opt.) vector that contains the difference image of each registered pair
 * @return
 */
Label Aligner::alignPair(const std::vector<cv::Mat>& _magicIms, int idx, double* convLogLikelihoodPtr)
{

    std::vector<cv::Mat> magicIms(_magicIms);
    // first convert all images to double
    for (size_t i=0; i<magicIms.size(); ++i)
    {
        if (magicIms[i].type() != CV_64FC1)
            magicIms[i].convertTo(magicIms[i], CV_64FC1);
    }

    // register tuples of images

    std::vector<cv::Mat> segment, segmentCrp;

    for (int j=0; j<=1; ++j) {
        segment.push_back(magicIms[j]);
        segmentCrp.push_back(imutil::cropCore(magicIms[j], rfw::imWs[MC->getPatchType()]));
    }


    Label estLabel(0,0,1,0);
    cv::Mat imRegLast;


    double convLogLikelihood;
    for (size_t m=0; m<MC->models.size(); m++)
    {
        std::vector<cv::Mat> tmpSegment;
        for (size_t q=0; q<segment.size(); ++q) {
            tmpSegment.push_back(segment[q].clone());
        }

        cv::Mat tmppp = alignTmp(tmpSegment, m,  &convLogLikelihood, &estLabel);



        if (m == MC->models.size()-1)
        {
            imRegLast = segment.back().clone();



            imRegLast = imutil::shift(imRegLast, -estLabel.tx, -estLabel.ty);
            imRegLast = imutil::resizeUpscaled(imRegLast, 1./estLabel.sc);
            imRegLast = imutil::rotate(imRegLast, -estLabel.r);
        }
    }

    if (convLogLikelihoodPtr != nullptr)
        *convLogLikelihoodPtr = convLogLikelihood;
    /****
#ifdef VERB
    std::cout << "REGISTERED PAIR" << std::endl;
    std::cout << "REGISTERED PAIR" << std::endl;
    std::cout << "REGISTERED PAIR" << std::endl;

    for (size_t k=0; k<20; ++k) {
        cv::imshow("Before: \hat{I}_t, I_{t+1}",before1/255);
        cv::waitKey(50);
        cv::imshow("Before: \hat{I}_t, I_{t+1}",before2/255);
        cv::waitKey(50);
    }
    cv::imshow("Difference BEFORE", cvip::Image::doubleToUchar(diffBef));

    cim2 = imutil::cropCore(im2);

    cv::Mat diffAfter = cim2-cim1;

    for (size_t k=0; k<20; ++k) {
        cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim1/255);
        cv::waitKey(50);
        cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim2/255);
        cv::waitKey(50);
    }

    cv::imshow("Difference AFTER ", cvip::Image::doubleToUchar(diffAfter));
#endif
***/
    /************
    if (idx != -1)
    {
        size_t pad = 30;
        for (size_t i=0; i<pad; i++)
        {
            std::stringstream ssb1, ssb2, ssa1, ssa2;
            ssb1 << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_" << std::setfill('0') << std::setw(4) << i*2 << ".png";
            ssb2 << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_" << std::setfill('0') << std::setw(4) << i*2+1 << ".png";

            ssa1 << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_" << std::setfill('0') << std::setw(4) << pad+i*2 << ".png";
            ssa2 << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_" << std::setfill('0') << std::setw(4) << pad+i*2+1 << ".png";

            cv::imwrite(ssb1.str(), before1);
            cv::imwrite(ssb2.str(), before2);

            cv::imwrite(ssa1.str(), cim1);
            cv::imwrite(ssa2.str(), cim2);

    //        cv::imwrite(QString("visual/after_"+QString::number(idx)+"_1.png").toStdString(), cim1);
    //        cv::imwrite(QString("visual/after_"+QString::number(idx)+"_2.png").toStdString(), cim2);
        }
    }
********/
    return estLabel;
}













/**
 * @brief Aligner::align  align a sequence
 * @param magicIms
 * @param beforeDiffs - (opt.) vector that contains the difference image of each pair before registration
 * @param afterDiffs - (opt.) vector that contains the difference image of each registered pair
 * @return
 */
std::tuple<std::vector<cv::Mat>, std::vector<int> > Aligner::alignSequenceWithCorrection(const std::vector<cv::Mat>& _magicIms, std::vector<cv::Mat> *beforeDiffs,
                                                                                         std::vector<cv::Mat> *afterDiffs, std::vector<Label>* gtLabels, std::vector<Label>* estLabels,
                                                                                         std::vector<double> *convProbs, std::vector<cv::Mat> *estWarps, std::vector<bool> *successFlag,
                                                                                         std::vector<cv::Mat> *allEstWarps)
{

    std::vector<cv::Mat> segments;
    std::vector<cv::Mat> magicIms(_magicIms);

    std::vector<int> registrationStatus(magicIms.size(),0);

    int T = magicIms.size();

    // first convert all images to double
    for (size_t i=0; i<magicIms.size(); ++i)
    {
        if (magicIms[i].type() != CV_64FC1)
        {
            magicIms[i].convertTo(magicIms[i], CV_64FC1);
            if (operationScale != 1) {
                //                cv::resize(magicIms[i], magicIms[i], cv::Size(), operationScale, operationScale);
            }
        }
    }

    std::vector<std::vector<int> > framesToBeFixed(T, std::vector<int>());

    segments.push_back(magicIms[0]);
    successFlag->push_back(true); // first frame is assumed to be correctly registered
    registrationStatus[0] = 1;

    // The first label is the reference
    if (nullptr != estLabels)
        estLabels->push_back(Label(0,0,1,0));

    if (nullptr != estWarps)
    {
        Label ll(0,0,1,0);
        estWarps->push_back(ll.getWarpMatrix(magicIms[0]));
    }

    int lastInterruption = 0;

    int numConsecutiveFails = 0;


    // register tuples of images
    for (int i=1; i<magicIms.size(); ++i)
    {
        auto system_start = std::chrono::system_clock::now();
        //!std::cout << "Registering pair #" << i << " out of " << magicIms.size()-1 << "..." <<  std::endl;

        std::vector<cv::Mat> segment, segmentCrp;

        int idxStart = std::max<int>(lastInterruption, i-timeWindow+1);

        int tprime = i;
        std::vector<int> dummy;
        while (segment.size() < timeWindow && tprime >= 0)
        {
            // construct segment through successfully registered frames
            if (tprime == i || successFlag->at(tprime))
            {
                segment.insert(segment.begin(), magicIms[tprime]);
                segmentCrp.insert(segmentCrp.begin(), imutil::cropCore(magicIms[tprime], rfw::imWs[MC->getPatchType()]));
                dummy.push_back(tprime);
            }
            tprime--;
        }

        cv::Mat imRegLast;

        cv::Mat before1 = segmentCrp[segmentCrp.size()-2];
        cv::Mat before2 = segmentCrp[segmentCrp.size()-1];

        cv::Mat diffBef = before2-before1;

        if (nullptr != beforeDiffs) {
            beforeDiffs->push_back(diffBef);
        }

        // cumulative estimation, will be accumulated over cascaded models
        Label estLabel(0,0,1,0);

        double convProb = -1; // convergence probability

        for (int m=0; m<MC->models.size(); m++)
        {

            int64 t1 = cv::getTickCount();
            // the registered version of the last image in segment
            //            imRegLast = alignTmp(segment, m, &convProb, &estLabel, 0, i >= 9 && m==MC->models.size()-1 ); // i >= 16
            imRegLast = alignTmp(segment, m, &convProb, &estLabel, 0,true, allEstWarps); // i >= 16
            //!imRegLast = alignTmp(segment, m, &convProb, &estLabel, 0, false); // i >= 16

            if (m == MC->models.size()-1)
            {

                imRegLast = segment.back().clone();
                Label labelToApply(estLabel);
                imutil::applyLabel(imRegLast, labelToApply.invert());
            }
        }

        //! Now check
        std::vector<cv::Mat> controlSegment;
        controlSegment.push_back(segment[segment.size()-2].clone());
        controlSegment.push_back(imRegLast.clone());

        if (nullptr != thresholdMlp)
        {
            double convThreshold = 0.1*thresholdMlp->threshold;
            //!std::cout << "convProb" << convProb << " vs. " << convThreshold << std::endl;

            if (convProb < convThreshold && convProb != -1)
            {
                //!std::cout << " FAILURE DETECTED  " <<convProb<< '\t' << convThreshold;
                imRegLast = segment.back().clone();
                estLabel = Label(0,0,1,0);
                successFlag->push_back(false);
                registrationStatus[i] = 0;
                // attempt to correct this failure after Tw frames
                framesToBeFixed[std::min<int>(i+Tw,T-1)].push_back(i);

                numConsecutiveFails++;

                //! if we keep failing for a long time, it probably means that
                //! there was a large head pose variation or something
                //! we'll try to re-start registration with new reference frame
                if (numConsecutiveFails == 3) {
                    //!std::cout << " ... TO MANY FAILURES! RESTARTING FROM  NEW REFERENCE.... " <<convProb<< '\t' << convThreshold;
                    //!std::cout << " ... TO MANY FAILURES! RESTARTING FROM  NEW REFERENCE.... " <<convProb<< '\t' << convThreshold;
                    //!std::cout << " ... TO MANY FAILURES! RESTARTING FROM  NEW REFERENCE.... " <<convProb<< '\t' << convThreshold;

                    numConsecutiveFails = 0;
                    successFlag->at(successFlag->size()-1) = true;

                    //! reg. status of 3 means we restarted registration
                    //! at this frame...
                    registrationStatus[i] = 3;

                    //! since this is the new reference, forget the past mistakes
                    for (size_t ii=i; ii<framesToBeFixed.size(); ++ii)
                        framesToBeFixed[ii].clear();
                }
            } else {
                successFlag->push_back(true);
                registrationStatus[i] = 1;
                numConsecutiveFails = 0;
            }
        }
        else // we assume everything is correctly registered if no failure identification
            successFlag->push_back(true);

        if (nullptr != convProbs)
            convProbs->push_back(convProb);

        if (nullptr != estLabels)
            estLabels->push_back(estLabel);

        if  (nullptr != estWarps)
            estWarps->push_back(estLabel.getWarpMatrix(magicIms[i]));



        cv::Mat cim1After = imutil::cropCore(segment[segment.size()-2].clone(), rfw::imWs[MC->getPatchType()]);
        cv::Mat cim2After = imutil::cropCore(imRegLast.clone(), rfw::imWs[MC->getPatchType()]);

        cv::Mat diffAfter = cim2After-cim1After;

#ifdef EXTRAVERB
        std::cout << "REGISTERED PAIR" << std::endl;
        std::cout << "REGISTERED PAIR" << std::endl;
        std::cout << "REGISTERED PAIR" << std::endl;

        for (size_t k=0; k<20; ++k) {
            cv::imshow("Before: \hat{I}_t, I_{t+1}",before1/255);
            cv::waitKey(50);
            cv::imshow("Before: \hat{I}_t, I_{t+1}",before2/255);
            cv::waitKey(50);
        }
        cv::imshow("Difference BEFORE", cvip::Image::doubleToUchar(diffBef));

        for (size_t k=0; k<20; ++k) {
            cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim1After/255);
            cv::waitKey(50);
            cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim2After/255);
            cv::waitKey(50);
        }

        cv::imshow("Difference AFTER ", cvip::Image::doubleToUchar(diffAfter));
#endif

        if (nullptr != afterDiffs)
            afterDiffs->push_back(diffAfter);

        cv::Mat rawImLast = segment.back().clone();
        magicIms[i] = imRegLast;

        if (convProb < MC->getThreshold())
        {
            //!!!!!!!!!!!!!!!!!!
            //!!!!!!!!!!!!!!!!!!
            //!!!!!!!!!!!!!!!!!!
            //!!!!!!!!!!!!!!!!!!
            //!!!!!!!!!!!!!!!!!!
            //!!!!!!!!!!!!!!!!!!
            //!!!!!!!!!!!!!!!!!!
            //!!!!!!!!!!!!!!!!!!
            //!!!!!!!!!!!!!!!!!!
        }

        segments.push_back(magicIms[i]);

        auto diff = std::chrono::system_clock::now() - system_start;
        auto sec = std::chrono::duration_cast<std::chrono::milliseconds>(diff);

        if (framesToBeFixed[i].size()) {
            for (int tp = 0; tp<framesToBeFixed[i].size(); ++tp) {
                auto p = attemptCorrection(framesToBeFixed[i][tp], magicIms, estLabels, estWarps, successFlag);
                if (std::get<0>(p) == true)
                {
                    registrationStatus[framesToBeFixed[i][tp]] = 2;
                }
            }
        }

        //!std::cout << sec.count() << " milliseconds." << std::endl;
    }

    return std::make_tuple(magicIms, registrationStatus);
}












/**
 * @brief Aligner::align  align a sequence
 * @param magicIms
 * @param beforeDiffs - (opt.) vector that contains the difference image of each pair before registration
 * @param afterDiffs - (opt.) vector that contains the difference image of each registered pair
 * @return
 */
tuple<vector<Mat>, vector<bool>, vector<Mat> > Aligner::alignOnline(const std::vector<cv::Mat>& _magicIms, bool identifyFailure)
{

    vector<Mat> registered;
    vector<bool> success;
    vector<Mat> registeredWithSuccess;

    std::vector<cv::Mat> magicIms(_magicIms);

    // first convert all images to double
    for (size_t i=0; i<magicIms.size(); ++i)
    {
        if (magicIms[i].type() != CV_64FC1)
            magicIms[i].convertTo(magicIms[i], CV_64FC1);
    }

    registered.push_back(magicIms[0]);
    registeredWithSuccess.push_back(magicIms[0]);
    success.push_back(true);

    int lastSuccessIdx = 0;

    // register tuples of images
    for (int i=1; i<magicIms.size(); ++i)
    {
        auto system_start = std::chrono::system_clock::now();
        //!std::cout << "Registering pair #" << i << " out of " << magicIms.size()-1 << "..." ;

        std::vector<cv::Mat> segment, segmentCrp;

        int idxStart;

        if (identifyFailure) {
            idxStart = lastSuccessIdx;
        } else {
            idxStart = i-1;
        }

        segment.push_back(magicIms[idxStart]);
        segment.push_back(magicIms[i]);
        segmentCrp.push_back(imutil::cropCore(magicIms[idxStart], rfw::imWs[MC->getPatchType()]));
        segmentCrp.push_back(imutil::cropCore(magicIms[i], rfw::imWs[MC->getPatchType()]));

        cv::Mat imRegLast;

        cv::Mat before1 = segmentCrp[segmentCrp.size()-2];
        cv::Mat before2 = segmentCrp[segmentCrp.size()-1];

        cv::Mat diffBef = before2-before1;

        // cumulative estimation, will be accumulated over cascaded models
        Label estLabel(0,0,1,0);

        double convLogLikelihood, refLogLikelihood;

        for (int m=0; m<MC->models.size(); m++)
        {

            int64 t1 = cv::getTickCount();
            // the registered version of the last image in segment
            imRegLast = alignTmp(segment, m, &convLogLikelihood, &estLabel, 0, true);

            if (m == MC->models.size()-1)
            {
                imRegLast = segment.back().clone();
                imutil::applyLabel(imRegLast, estLabel);
            }
        }


        //! Now check convergence
        std::vector<cv::Mat> controlSegment;
        //        controlSegment.push_back(segment[segment.size()-2].clone());
        controlSegment.push_back(magicIms[idxStart].clone());
        controlSegment.push_back(imRegLast.clone());

        refLogLikelihood = computeConvergenceLikelihood(controlSegment, MC->modelForThreshold);

        // perform back transformation only if we think registration is good enough
        if (refLogLikelihood < MC->getThreshold())
        {
            std::cout << " FAILURE DETECTED \t" << MC->getThreshold() << " vs. " << refLogLikelihood << '\t' << MC->getUniqueKey();
            imRegLast = segment.back().clone();
            estLabel = Label(0,0,1,0);
            success.push_back(false);
        } else {
            lastSuccessIdx = i;
            success.push_back(true);
        }

        cv::Mat cim1After = imutil::cropCore(segment[segment.size()-2].clone(), rfw::imWs[MC->getPatchType()]);
        cv::Mat cim2After = imutil::cropCore(imRegLast.clone(), rfw::imWs[MC->getPatchType()]);

        cv::Mat diffAfter = cim2After-cim1After;

#ifdef EXTRAVERB
        std::cout << "REGISTERED PAIR" << std::endl;
        std::cout << "REGISTERED PAIR" << std::endl;
        std::cout << "REGISTERED PAIR" << std::endl;

        for (size_t k=0; k<20; ++k) {
            cv::imshow("Before: \hat{I}_t, I_{t+1}",before1/255);
            cv::waitKey(50);
            cv::imshow("Before: \hat{I}_t, I_{t+1}",before2/255);
            cv::waitKey(50);
        }
        cv::imshow("Difference BEFORE", cvip::Image::doubleToUchar(diffBef));

        for (size_t k=0; k<20; ++k) {
            cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim1After/255);
            cv::waitKey(50);
            cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim2After/255);
            cv::waitKey(50);
        }

        cv::imshow("Difference AFTER ", cvip::Image::doubleToUchar(diffAfter));
#endif

        cv::Mat rawImLast = segment.back().clone();
        registered.push_back(imRegLast);
        if (true == success.back()) {
            registeredWithSuccess.push_back(imRegLast);
        }

        magicIms[i] = imRegLast;

        auto diff = std::chrono::system_clock::now() - system_start;
        auto sec = std::chrono::duration_cast<std::chrono::milliseconds>(diff);

        std::cout << sec.count() << " milliseconds." << std::endl;
    }

    return std::make_tuple(registered, success, registeredWithSuccess);
}




/**
 * @brief Aligner::align  align a sequence
 * @param magicIms
 * @param beforeDiffs - (opt.) vector that contains the difference image of each pair before registration
 * @param afterDiffs - (opt.) vector that contains the difference image of each registered pair
 * @return
 */
std::pair<Label, bool> Aligner::alignPair(std::tuple<cv::Mat, cv::Mat> &pair, double* convLogLikelihoodPtr, int idx)
{
    cv::Mat im1 = std::get<0>(pair).clone();
    cv::Mat im2 = std::get<1>(pair).clone();

    if (im1.channels()>1)
        cv::cvtColor(im1, im1, cv::COLOR_RGB2GRAY);

    if (im2.channels()>1)
        cv::cvtColor(im2, im2, cv::COLOR_RGB2GRAY);

    im1.convertTo(im1,CV_64FC1);
    im2.convertTo(im2,CV_64FC1);

    // register tuples of images

    cv::Mat cim1 = imutil::cropCore(im1, rfw::imWs[MC->getPatchType()]);
    cv::Mat cim2 = imutil::cropCore(im2, rfw::imWs[MC->getPatchType()]);


    cv::Mat before1 = cim1.clone();
    cv::Mat before2 = cim2.clone();
    cv::Mat diffBef = before2-before1;

    Label estLabel(0,0,1,0);

#ifdef EXTRAVERB
    for (size_t k=0; k<20; ++k) {
        cv::imshow("Before: \hat{I}_t, I_{t+1}",before1/255);
        cv::waitKey(50);
        cv::imshow("Before: \hat{I}_t, I_{t+1}",before2/255);
        cv::waitKey(50);
    }
#endif

    double convLogLikelihood;
    double refLogLikelihood;
    for (size_t m=0; m<MC->models.size(); m++)
    {
        //        auto tmpp = std::make_tuple(im1,im2);
        std::vector<cv::Mat> tmpp;
        tmpp.push_back(im1);
        tmpp.push_back(im2);

        cv::Mat truelyTmp = alignTmp(tmpp, m,  &convLogLikelihood, &estLabel,0,true);

        /*********
        if (m == MC->thresholdRefIdx) {
            refLogLikelihood = convLogLikelihood;
        }
        *********/

    }

    im1 = im1.clone();
    im2 = std::get<1>(pair).clone();

    Label labelToApply(estLabel);
    imutil::applyLabel(im2, labelToApply.invert());



    im2.convertTo(im2,CV_64FC1);

    std::vector<cv::Mat> controlSegment;
    controlSegment.push_back(im1);
    controlSegment.push_back(im2);

    pair = std::make_tuple(im1.clone(), im2.clone());

    cim2 = imutil::cropCore(im2, rfw::imWs[MC->getPatchType()]);
    pair = std::make_tuple(cvip::Image::doubleToUchar(cim1), cvip::Image::doubleToUchar(cim2));

#ifdef EXTRAVERB
    before1  = imutil::cropCore(before1, rfw::imWs[MC->getPatchType()]*operationScale);
    before2  = imutil::cropCore(before2, rfw::imWs[MC->getPatchType()]*operationScale);

    cim1 = imutil::cropCore(cim1, rfw::imWs[MC->getPatchType()]*operationScale);
    cim2 = imutil::cropCore(cim2, rfw::imWs[MC->getPatchType()]*operationScale);

    for (size_t k=0; k<20; ++k) {
        cv::imshow("Before: \hat{I}_t, I_{t+1}",before1/255);
        cv::waitKey(50);
        cv::imshow("Before: \hat{I}_t, I_{t+1}",before2/255);
        cv::waitKey(50);
    }
    cv::imshow("Difference BEFORE", cvip::Image::doubleToUchar(diffBef));

    cv::Mat diffAfter = cim2-cim1;

    for (size_t k=0; k<20; ++k) {
        cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim1/255);
        cv::waitKey(50);
        cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim2/255);
        cv::waitKey(50);
    }

    cv::imshow("Difference AFTER ", cvip::Image::doubleToUchar(diffAfter));
#endif

    if (idx != -1)
    {
        size_t pad = 8;

        before1 = before1(cv::Rect(20,20,160,160));
        before2 = before2(cv::Rect(20,20,160,160));

        cim1 = cim1(cv::Rect(20,20,160,160));
        cim2 = cim2(cv::Rect(20,20,160,160));

        cv::Mat diffb = before2-before1;
        diffb.convertTo(diffb, CV_32F);
        cv::pow(diffb,2,diffb);

        cv::Mat diffa = cim2-cim1;
        diffa.convertTo(diffa, CV_32F);
        cv::pow(diffa,2,diffa);

        std::stringstream ssdb,ssda;
        ssdb << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_diff_before.png";
        ssda << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_diff_after.png";

        diffb = 255-cvip::Image::doubleToUchar(diffb);
        diffa = 255-cvip::Image::doubleToUchar(diffa);
        cv::imwrite(ssdb.str(), diffb);
        cv::imwrite(ssda.str(), diffa);


        for (size_t i=0; i<pad; i++)
        {
            std::stringstream ssb1, ssb2, ssa1, ssa2;
            ssb1 << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_" << std::setfill('0') << std::setw(4) << i*2 << ".png";
            ssb2 << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_" << std::setfill('0') << std::setw(4) << i*2+1 << ".png";

            ssa1 << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_" << std::setfill('0') << std::setw(4) << 2*pad+i*2 << ".png";
            ssa2 << rfw::VISUAL_PAIRS_PATH << '/' << idx << "_" << std::setfill('0') << std::setw(4) << 2*pad+i*2+1 << ".png";

            cv::imwrite(ssb1.str(), before1);
            cv::imwrite(ssb2.str(), before2);

            cv::imwrite(ssa1.str(), cim1);
            cv::imwrite(ssa2.str(), cim2);

            //        cv::imwrite(QString("visual/after_"+QString::number(idx)+"_1.png").toStdString(), cim1);
            //        cv::imwrite(QString("visual/after_"+QString::number(idx)+"_2.png").toStdString(), cim2);
        }
    }

    /***************
    if (refLogLikelihood > MC->getThreshold())
    {
        std::cout << refLogLikelihood << std::endl;
        std::cout << refLogLikelihood << std::endl;
        std::cout << refLogLikelihood << std::endl;
        std::cout << refLogLikelihood << std::endl;
        std::cout << refLogLikelihood << std::endl;
        for (size_t k=0; k<200; ++k) {
            cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim1/255);
            cv::waitKey(50);
            cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",cim2/255);
            cv::waitKey(50);
        }
    }
****************/
    return std::make_pair(estLabel, true);
}


/**
 * @brief Aligner::isOscillating - identify a possible oscillation in the process of registration (i.e. are we going back and forth between two estimations?
 * @param labels
 * @return
 */
bool Aligner::isOscillating(const std::vector<size_t> &labels, size_t allowedRepetitions) const
{
    // wee need this many labels two see if we are alternating between 2 labels
    if (labels.size() < 2*allowedRepetitions)
        return false;

    // last labels
    std::vector<size_t> llabels;
    std::copy(labels.end()-2*allowedRepetitions, labels.end(), std::back_inserter(llabels));

    size_t l1 = llabels[0];
    size_t l2 = llabels[1];
    for (size_t i=2; i<llabels.size(); i+=2)
    {
        if (l1!= llabels[i])
            return false;
    }

    for (size_t i=3; i<llabels.size(); i+=2)
    {
        if (l2!= llabels[i])
            return false;
    }

    return true;
}

/**
 * @brief Aligner::findReferenceImage - Given a sequence, try to find a reference image to register the remaining images to (I_r).
 * @param origSeq - the sequence
 * @param rawLabels - (opt) if we have the labels (Label(dtx,dty,dsc)) of images, use for comparison -- mostly training
 * @return
 */
size_t Aligner::findReferenceImage(const std::vector<cv::Mat> &origSeq, const std::vector<Label> *rawLabels) const
{
    std::vector<double> totalCosts(origSeq.size(),0.);
    std::vector<double> totalProbs(origSeq.size(),0.);

    std::vector<cv::Mat> seq;

    // copy sequence, don't touch the original
    for (size_t i=0; i<origSeq.size(); ++i)
    {
        cv::Mat tmp = origSeq[i];
        tmp.convertTo(tmp, CV_64FC1);
        seq.push_back(tmp);
    }

    // use first model for estimation
    const Model* M = &MC->models[1];

    for (size_t i=0; i<seq.size(); ++i)
    {
        Label al1(rawLabels->at(i));
        std::cout << "Checking t=" << i << "\t\t" << al1.tx << "\t" << al1.ty << "\t" << al1.sc << std::endl;
        std::cout << "========================================================= " << std::endl;

        for (size_t j=0; j<seq.size(); ++j)
        {
            if (i==j)
                continue;

            cv::Mat tmpi = imutil::cropCore(seq[i], rfw::imWs[MC->getPatchType()]);
            cv::Mat tmpj = imutil::cropCore(seq[j], rfw::imWs[MC->getPatchType()]);


            double prob, convProb;
            cvip::Mat3 mat(tmpi, tmpj);
            int estLabelIx = M->estimateLabelNB2(mat, &prob, &convProb);

            Label al2(rawLabels->at(j));

            Label l(estLabelIx, M);

            using std::pow;
            double dt = std::sqrt(pow(l.tx,2.)+pow(l.ty,2.)+pow(2*l.sc*rfw::imW,2.));
            // std::cout << "Total cost: " << dt << "\t" << convProb << "\t" << al1.tx-al2.tx << "\t" << al1.ty-al2.ty << "\t" << al1.sc-al2.sc << "\t" << std::endl;

            totalCosts[i] += dt;
            totalProbs[i] += convProb;

#ifdef GUI
            for (size_t k=0; k<0; ++k)
            {
                cv::imshow("refim", tmpi/255);
                cv::waitKey(50);
                cv::imshow("refim", tmpj/255);
                cv::waitKey(50);
            }
#endif
        }
        std::cout << "========================================================= " << std::endl;

    }


    for (size_t i=0; i<totalCosts.size(); ++i)
    {
        Label rl(rawLabels->at(i));
        std::cout << i << ") " << totalCosts[i]/totalCosts.size() << "\t" << totalProbs[i]/totalCosts.size() << '\t' << rl.tx << '\t' << rl.ty << '\t' << rl.sc << std::endl;
    }

    return 0;

}

/**
 * @brief Aligner::align -- align a pair of images
 * @param magicIms - image pair
 * @param mIdx - index of model (MC->models[mIdx])
 * @param cumEst - this function is called separately for each model (for the same pair), so there is a cumulated estimation over models
 * @return
 */
cv::Mat Aligner::alignTmp(const std::vector<cv::Mat> &magicIms, int mIdx, double* convProb, Label *cumEst, size_t _MAX_IT, bool greenLight, std::vector<cv::Mat> *allEstWarps)
{
    bool converged = false;
    bool maxItReached = false;

    size_t i=0;

    const Model* M = &MC->models[mIdx];
    bool isFirstModel = mIdx == 0;
    bool isLastModel = mIdx == MC->models.size()-1;
    bool isLastTwoModels = mIdx == MC->models.size()-1 || mIdx == MC->models.size()-2;

    size_t MAX_IT = 8;
    //    if (mIdx == 0)
    //        MAX_IT = 10;
    if (isLastModel) {
        MAX_IT = 10;
    }

    if (_MAX_IT != 0)
        MAX_IT = _MAX_IT;

    //    MAX_IT = 0;


    // keep probabilities here so if cannot converge, find best convergence probability
    std::vector<double> convProbs; // relative convergence probabilities
    std::vector<Label> labels;

    cv::Mat tmpIm = magicIms.back().clone();

    std::vector<cv::Mat> tmpSegmentCrp;
    for (size_t i=0; i<magicIms.size(); ++i)
        tmpSegmentCrp.push_back(imutil::cropCore(magicIms[i].clone(), rfw::imWs[MC->getPatchType()]*operationScale));

    //! prepend segment with images if needed
    if (M->F->getTimeWindow() > tmpSegmentCrp.size())
    {
        std::vector<cv::Mat> _tmpSegmentCrp(tmpSegmentCrp);
        tmpSegmentCrp.clear();

        // prepend with the first image in the segment
        for (size_t i=0; i<(M->F->getTimeWindow()-_tmpSegmentCrp.size()); ++i )
            tmpSegmentCrp.push_back(_tmpSegmentCrp[0].clone());

        for (size_t i=0; i<_tmpSegmentCrp.size(); ++i)
            tmpSegmentCrp.push_back(_tmpSegmentCrp[i].clone());
    }


    if (tmpSegmentCrp.size() > M->F->getTimeWindow()) {
        int fdiff = tmpSegmentCrp.size()-M->F->getTimeWindow();

        tmpSegmentCrp.erase(tmpSegmentCrp.begin(), tmpSegmentCrp.begin()+fdiff);
    }


    if (M->F->getTimeWindow() == 2 && tmpSegmentCrp.size() > 2)
    {
        std::vector<cv::Mat> _tmpSegmentCrp(tmpSegmentCrp);
        tmpSegmentCrp.clear();

        int T = _tmpSegmentCrp.size();

        tmpSegmentCrp.push_back(_tmpSegmentCrp[T-2].clone());
        tmpSegmentCrp.push_back(_tmpSegmentCrp[T-1].clone());
    }

    Label labelToApply = cumEst->invert();


    //!!!
    imutil::applyLabel(tmpIm, labelToApply, operationScale);

    cv::Mat lastCroppedIm = tmpSegmentCrp[tmpSegmentCrp.size()-2].clone();

    std::vector<size_t> pastLabelIdx;


    while (!converged)
    {
        if (numIterations != -1)
        {
            if (performedIterations >= numIterations )
                break;

            performedIterations++;
        }



        if (!MAX_IT)
            break;

        tmpSegmentCrp[tmpSegmentCrp.size()-1] = imutil::cropCore(tmpIm, rfw::imWs[MC->getPatchType()]*operationScale);

        // estimate and save probability of estimation

        std::vector<double> features = M->F->computeFeatures(tmpSegmentCrp);
        double energy = M->F->signalEnergy(features);
        cv::Mat featuresMat = cvip::Image::vectorToColumn(features);


        //        convProb = this->MC->modelForThreshold->


        // PUT BACK!-!
        //        std::vector<double> estimated = mdnPtr->predict(featuresMat, &prob);

        //        FeatureExtractor::drawMap(mdnPtr->selectedFeats);


        double dummy;
        int clusterIdx;
        std::vector<double> estimated = M->mdnPtr->predict(featuresMat, energy, &dummy, &clusterIdx);
        Label l(estimated[0], estimated[1], estimated[2], estimated[3]);

        //        l.sc /= 10;

        labels.push_back(Label(*cumEst));

        bool converged = false;
        bool surelyConverged = false;
        bool hasToHaveConverged = false;
        if (nullptr != thresholdMlp && nullptr != thresholdF)
        {
            // estimate conv. prob ONLY if we are at the right feature extraction scheme.
            if (M->F->getUniqueKey() == thresholdF->getUniqueKey())
            {
                std::vector<cv::Mat> tmpSegmentCrpPair;
                tmpSegmentCrpPair.push_back(tmpSegmentCrp[tmpSegmentCrp.size()-2]);
                tmpSegmentCrpPair.push_back(tmpSegmentCrp[tmpSegmentCrp.size()-1]);

                std::vector<double>cMatVec = thresholdF->computeFeatures(tmpSegmentCrpPair);
                cv::Mat cMat= cvip::Image::vectorToColumn(cMatVec);
                std::vector<double> tmp = thresholdMlp->predict(cMat);

                *convProb = thresholdMlp->sigmoid(tmp[0]);

                surelyConverged = (*convProb > 1.3*thresholdMlp->threshold);
                hasToHaveConverged = (*convProb > 2.6*thresholdMlp->threshold);
            }
            //            std::cout << converged << '\t' << confidence << std::endl;
        }

        convProbs.push_back(*convProb);


        /*
        if (fabs(l.r) < 0.000001) l.r = 0.;
        if (fabs(l.sc) < 0.000001) l.sc = 0.;
        */


        if (fabs(l.r) < 0.000001) l.r = 0.;
        if (fabs(l.sc) < 0.000001) l.sc = 0.;



        double threshold = 0.0025; // put back 0.0005
        /*
        if (!isLastModel)
            threshold = 0.5;
    `   */
        if (l.avgErr() < threshold) // put back 0.0005
        {
            //!std::cout << "Stop overoptimizing!" << std::endl;
            break;
        }

        cumEst->tx += l.tx;
        cumEst->ty += l.ty;
        cumEst->sc *= (1+l.sc);
        cumEst->r += l.r;


        if (nullptr != allEstWarps)
        {
            cv::Mat H = cumEst->invert().getWarpMatrix(tmpIm);
            allEstWarps->push_back(H);
        }
        /*
        */
        if (surelyConverged && !isLastModel)
        {
#ifdef VERB
            std::cout << "Converged at early layer, proceeding with the next layer" << std::endl;
#endif VERB
            break;
        }

        if (isLastModel)
        {
#ifdef VERB
            //            std::cout << "It has to have converged :" << confidence << std::endl;
#endif VERB
            int asdadas=1;
            Label labelToApply = cumEst->invert();

            tmpIm = magicIms.back().clone();
            imutil::applyLabel(tmpIm, labelToApply);
            cv::Mat cim2 = imutil::cropCore(tmpIm.clone(),rfw::imWs[MC->getPatchType()]*operationScale);

#ifdef EXTRAVERB
            if (greenLight)
            {
                cv::Mat tmp1 = lastCroppedIm.clone();
                cv::Mat tmp2 = cim2.clone();

                //                cv::resize(tmp1, tmp1, cv::Size(), 3, 3);
                //                cv::resize(tmp2, tmp2, cv::Size(), 3, 3);

                for (size_t k=0; k<4; ++k) {
                    cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",tmp1/255);
                    cv::waitKey(50);
                    cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",tmp2/255);
                    cv::waitKey(50);
                }
            }
#endif
            //            break;
        }


        Label labelToApply = cumEst->invert();
        //        Label labelToApply(*cumEst);


#ifdef VERB
        std::cout << std::setw(2) << std::right << std::setfill('0') << i+1 << std::setfill(' ') << ") tx=" << std::setw(7) << l.tx << "\t ty=" << std::setw(7) << l.ty << "\t sc=" << std::setw(7) << std::setprecision(8) << l.sc << " r="
                  << std::setw(6) << std::left << l.r
                  << " (cluster Idx: " << clusterIdx << ", model " << mIdx+1 << "/" << MC->models.size() << ")" <<  std::endl;
#endif

        tmpIm = magicIms.back().clone();
        imutil::applyLabel(tmpIm, labelToApply,operationScale);

        cv::Mat cim2 = imutil::cropCore(tmpIm.clone(),rfw::imWs[MC->getPatchType()]*operationScale);

        if (0.00001 > fabs(l.sc) && fabs(l.tx) < 0.00001 && fabs(l.ty) < 0.00001 && fabs(l.r) < 0.00001)
            break;

#ifdef EXTRAVERB
        if (greenLight)
        {
            cv::Mat tmp1 = lastCroppedIm.clone();
            cv::Mat tmp2 = cim2.clone();


            for (size_t k=0; k<4; ++k) {
                cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",tmp1/255);
                cv::waitKey(50);
                cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",tmp2/255);
                cv::waitKey(50);
            }
        }
#endif
        ++i;




        if (isOscillating(pastLabelIdx))
        {
#ifdef VERB
            std::cout << " \t ===> stopping oscillation..." << std::endl;
#endif
            i = MAX_IT;
        }

        if (i >= MAX_IT)
        {
            //// PUT BACK!-!
            maxItReached = true;
            break;
        }
    }




    if (isLastModel  && nullptr != thresholdMlp && nullptr != thresholdF)
    {
        std::vector<cv::Mat> tmpSegmentCrpPair;
        tmpSegmentCrpPair.push_back(tmpSegmentCrp[tmpSegmentCrp.size()-2]);
        tmpSegmentCrpPair.push_back(tmpSegmentCrp[tmpSegmentCrp.size()-1]);

        std::vector<double>cMatVec = thresholdF->computeFeatures(tmpSegmentCrpPair);
        cv::Mat cMat= cvip::Image::vectorToColumn(cMatVec);
        std::vector<double> tmp = thresholdMlp->predict(cMat);

        *convProb = thresholdMlp->sigmoid(tmp[0]);
    }



    return tmpIm;
}




/**
 * @brief Aligner::align -- align a pair of images
 * @param magicIms - image pair
 * @param mIdx - index of model (MC->models[mIdx])
 * @param cumEst - this function is called separately for each model (for the same pair), so there is a cumulated estimation over models
 * @return
 */
double Aligner::computeConvergenceLikelihood(const std::vector<cv::Mat> &magicIms, const Model *modelPtr) const
{
    bool converged = false;
    bool maxItReached = false;

    size_t i=0;

    const Model* M = modelPtr;


    // keep probabilities here so if cannot converge, find best convergence probability
    std::vector<double> probs, convProbs, relConvProbs; // relative convergence probabilities
    std::vector<Label> labels;

    cv::Mat tmpIm = magicIms.back().clone();

    std::vector<cv::Mat> tmpSegmentCrp;
    for (size_t i=0; i<magicIms.size(); ++i)
        tmpSegmentCrp.push_back(imutil::cropCore(magicIms[i].clone(), rfw::imWs[MC->getPatchType()]));

    /**
    for (size_t k=0; k<40; ++k) {
        cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",tmpSegmentCrp[0]/255);
        cv::waitKey(40);
        cv::imshow("\\hat{I}_t, \\til{I}_{t+1}",tmpSegmentCrp[1]/255);
        cv::waitKey(40);
    }
**/

    // estimate and save probability of estimation
    double prob, convProb, totProb;

    int estLabelIx = M->estimateLabelNB2(tmpSegmentCrp, &prob, &convProb, &totProb);

    return convProb;
}



void Aligner::saveRegisteredVideo(const std::string& path, const std::string& patchType,  vector<Mat> &before, vector<Mat> &after,
                                  std::vector<cv::Mat> *estWarps,  std::vector<int> *registrationStatus, std::vector<std::vector<double> >* poses,
                                  std::vector<cv::Mat> *landmarks, bool saveBefore, const std::string& vidOutPath,  const std::string& successOutPath, int FPS, const std::string& vidBeforeOutPath) const
{
    // pahts to save (before, after)
    std::stringstream sspb,sspa;
    sspb << path << "-before_";
    sspa << path;

    if (registrationStatus->size() <= 0)
        return;

    cv::VideoWriter vidOut;
    cv::VideoWriter vidBefore;

    if (!vidOut.isOpened())
        vidOut.open(vidOutPath, cv::VideoWriter::fourcc('D','I','V','X'), FPS, cv::Size(rfw::imWs[patchType]*operationScale,rfw::imWs[patchType]*operationScale), true);

    if (!vidBefore.isOpened() && saveBefore)
        vidBefore.open(vidBeforeOutPath, cv::VideoWriter::fourcc('D','I','V','X'), FPS, cv::Size(rfw::imWs[patchType]*operationScale,rfw::imWs[patchType]*operationScale), true);

    std::vector<int> success;

    for (size_t j=0; j<before.size(); ++j)
    {
        success.push_back(registrationStatus->at(j));

        if (j<after.size()){
            cv::Mat frameAfter = Image::doubleToUcharNoNorm(imutil::cropCore(after[j], rfw::imWs[patchType]*operationScale));
            cv::cvtColor(frameAfter, frameAfter, cv::COLOR_GRAY2RGB);
            vidOut << frameAfter;

        } else {
            cv::Mat blank(rfw::imWs[patchType]*operationScale, rfw::imWs[patchType]*operationScale, CV_8U, cv::Scalar::all(255));
            vidOut << blank;
        }

        if (saveBefore) {
            cv::Mat tmpIm = imutil::cropCore(before[j], rfw::imWs[patchType]*operationScale);
            cv::cvtColor(tmpIm, tmpIm, cv::COLOR_GRAY2BGR);
            vidBefore << tmpIm;
        }
    }

    cvip::Image::writeVectorToFile(success, successOutPath);
    return;
}




void Aligner::saveOutput(const std::string& db, const std::string& patchType,  size_t i, vector<Mat> &before, vector<Mat> &after,
                         vector<Mat>* diffsBefore, vector<Mat>* diffsAfter, std::vector<cv::Point2f> *landmarks,
                         std::vector<cv::Mat> *estWarps, std::vector<bool> *successFlag, std::vector<int> *registrationStatus, const std::string& extraText) const
{
    /*********************
     * @@@@@@
     *
     *
    if (!QDir(rfw::VISUAL_PATH.c_str()).exists())
        QDir().mkdir(rfw::VISUAL_PATH.c_str());

    if (!QDir(rfw::VISUAL_SEQS_PATH.c_str()).exists())
        QDir().mkdir(rfw::VISUAL_SEQS_PATH.c_str());
    */
    std::string path(rfw::VISUAL_SEQS_PATH+"/"+this->MC->getUniqueKey()+extraText);

    /***********
     * @@@@@@@@@@@@@@@@@@
    if (!QDir(path.c_str()).exists())
        QDir().mkdir(path.c_str());

        */

    std::stringstream sspb,sspa;
    sspb << path << "/" << db << "-before_";// << std::setw(4) << std::setfill('0') << i;
    sspa << path << "/" << db;// << "-after_" << std::setw(4) << std::setfill('0') << i;
    /************
     * @@@@@@@@
    if (!QDir(sspb.str().c_str()).exists())
        QDir().mkdir(sspb.str().c_str());

    if (!QDir(sspa.str().c_str()).exists())
        QDir().mkdir(sspa.str().c_str());
*/

    if (nullptr != landmarks) {
        cv::Mat pts(landmarks->size(),2,CV_64FC1,cv::Scalar::all(0));
        for (size_t i=0; i<landmarks->size(); ++i) {
            pts.at<double>(i,0) = landmarks->at(i).x;
            pts.at<double>(i,1) = landmarks->at(i).y;
        }

        std::string path(sspa.str()+"/landmarks.txt");
        cvip::Image::writeToFile(pts,path);
    }

    for (size_t j=0; j<before.size(); ++j)
    {
        if (!successFlag->at(j))
        {
            //            continue;
        }
        std::stringstream ss, ssfull, ssdat, ssafterfull, ssrs;

        ss << sspb.str() <<  "/" <<  std::setw(6) << std::setfill('0') << j+1 << ".png";
        ssfull << sspb.str() <<  "/" <<  std::setw(4) << std::setfill('0') << j+1 << ".png";
        ssdat << sspa.str() <<  "/" <<  std::setw(6) << std::setfill('0') << j+1 << ".T";
        ssrs << sspa.str() <<  "/" <<  std::setw(6) << std::setfill('0') << j+1 << ".RS";
        cv::imwrite(ss.str(), imutil::cropCore(before[j], rfw::imWs[patchType]));
        cv::imwrite(ssfull.str(), Image::doubleToUcharNoNorm(imutil::cropCore(before[j], 170)));

        if (nullptr != registrationStatus)
            cvip::Image::writeValueToFile(registrationStatus->at(j), ssrs.str());


        if (nullptr != estWarps)
        {
            cvip::Image::writeToFile(estWarps->at(j), ssdat.str());
        }

        ss.str("");
        ss << sspa.str() <<  "/" <<  std::setw(6) << std::setfill('0') << j+1 << ".png";
        ssafterfull << sspa.str() <<  "/" <<  std::setw(4) << std::setfill('0') << j+1 << ".png";

        if (j<after.size()){
            cv::imwrite(ss.str(), Image::doubleToUcharNoNorm(imutil::cropCore(after[j], rfw::imWs[patchType]*operationScale)));
            //            cv::imwrite(ssafterfull.str(), Image::doubleToUcharNoNorm(imutil::cropCore(after[j], 170)));
        }

        if (diffsBefore != nullptr && diffsAfter != nullptr )
            if (j<diffsBefore->size() && diffsBefore->size())
            {
                ss.str("");
                ss << path << "/" << db << "_dbefore_" << std::setw(4) << std::setfill('0') << i <<  "_" << std::setw(6) << std::setfill('0') << j+1 << ".png";
                cv::Mat tmp = imutil::cropCore(diffsBefore->at(j), rfw::imWs[patchType]*operationScale);
                tmp.convertTo(tmp, CV_32F);
                cv::pow(tmp,2,tmp);
                cv::imwrite(ss.str(), 255-Image::doubleToUcharNoNorm(tmp));

                ss.str("");
                ss << path  << "/" << db<< "_dafter_" << std::setw(4) << std::setfill('0') << i << "_" << std::setw(6) << std::setfill('0') << j+1 << ".png";

                tmp = diffsAfter->at(j);
                tmp.convertTo(tmp, CV_32F);
                cv::pow(tmp,2,tmp);
                cv::imwrite(ss.str(), 255-Image::doubleToUcharNoNorm(tmp));
            }
    }
    return;
}



template <class T>
int Aligner::getSegmentOffset(const std::vector<std::vector<T> >& vec, int idx)
{
    int offset = 0;

    for (size_t i=0; i<idx; ++i)
        offset += vec[i].size();

    return offset;
}
