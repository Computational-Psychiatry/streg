#include <iostream>
#include <vector>

#include <experimental/filesystem>

#include "Image.hpp"
#include "RegFw.hpp"
#include "Utility.hpp"
#include "GaborBank.hpp"
#include "Models.hpp"
#include "Aligner.hpp"
#include "OpticalFlow.hpp"
#include "MachineLearning.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <stdio.h>

using namespace cvip;
using std::vector; using cv::Mat;

using namespace rfw;
using std::vector; using std::tuple; using std::string; using std::make_tuple;

std::vector<std::tuple<FeatureExtractor*, std::string, MultiMDN*> > modelParams, modelParams2;


void registerVideo(const std::string& videoPath, const std::string& patchType, const std::string &outDir, const int timeWindow, const bool saveBefore);


int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Insufficient number of parameters! You must pass the three following parameters: " << std::endl;
        std::cerr << "%videoPath %patchType %outdir (e.g. video1.mp4 leye ./output_dir)" << std::endl;
        return -1;
    }

    const std::string videoPath = argv[1];
    const std::string patchType = argv[2];
    const std::string outDir = argv[3];
    int Tw = 3;
    bool save_before = false;

    if (argc >= 5)
        Tw = std::atoi(argv[4]);

    if (argc >= 6)
        save_before= (bool) std::atoi(argv[5]);

    if (patchType != "leye" && patchType != "reye" && patchType != "mouth"){
        std::cerr << "Second argument (i.e. patchType) must be one of the following: leye, reye, mouth" << std::endl;
        return -1;
    }

    registerVideo(videoPath, patchType, outDir, Tw, save_before);

    return 0;
}



void registerVideo(const std::string& videoPath, const std::string& patchType, const std::string& outDir, const int timeWindow, const bool saveBefore)
{
    std::vector<std::tuple<FeatureExtractor*, std::string, MultiMDN*> > mp;
    OpticalFlow OF("optical_flow-5-15");

    GaborBank G34("3-4-4-10000-11111111-std-double");
    GaborBank G24("2-4-4-10000-11111111-std-double");

    MultiMDN mdnOF("data/mdnmodels/model_OF_"+patchType);
    MultiMDN mdnG34("data/mdnmodels/model_G34_"+patchType);

    mp.push_back(make_tuple(&OF,  patchType+"-8-0.31-1-1-0.01-0.005-2-1-5", &mdnOF));
    mp.push_back(make_tuple(&OF,  patchType+"-8-0.31-2-1-0.01-0.005-2-1-5", &mdnOF));
    mp.push_back(make_tuple(&G24, patchType+"-8-0.31-2-1-0.01-0.005-2-1-5", &mdnG34));
    mp.push_back(make_tuple(&G34, patchType+"-8-0.31-2-1-0.01-0.005-2-1-5", &mdnG34));

    ModelsCascade mc(mp);

    Aligner A(&mc);

    MLP thresholdMlp("data/mdnmodels/model_failure_"+patchType);
    A.thresholdMlp = &thresholdMlp;
    A.thresholdF = &G24;

    // initialize landmark detector
    vector<string> arguments(1, "null");

    int FPS = -1;
    {
        cv::VideoCapture cap(videoPath);
        FPS = cap.get(cv::CAP_PROP_FPS);
    }

    int Twidth = timeWindow*FPS;

    //! we'll use this rect to crop the face. It will be the same rect throughout the sequence.
    //! Obviously the head can move throughout the sequence, but we'll make it large enough to ensure that
    //! the rect is large enough so that the face is always in the cropped video
    cv::Rect faceRect(-1,-1,-1,-1);

    int t = -1;
    while (true)
    {
        t++;
        int tBegin = t*Twidth;
        int tEnd = tBegin+Twidth;


        // Create output folders to save
        std::stringstream ss, ssFullFace, ssOutVidPath, ssOutTxtPath, ssBeforeVidPath;
        ssFullFace << ss.str();

        ss << "/" << patchType << "-part-" << std::setw(4) << std::setfill('0') << t;

        // Save also full faces so we can later visualize them
        ssFullFace << "/fullface-" << std::setw(4) << std::setfill('0') << t;

        ssOutVidPath << outDir << "/" << std::experimental::filesystem::path(videoPath).stem().generic_string() << "-" << patchType << "-" << std::setw(4) << std::setfill('0') << t << ".avi";
        ssBeforeVidPath << outDir << "/" << std::experimental::filesystem::path(videoPath).stem().generic_string() << "-" << patchType << "-" << std::setw(4) << std::setfill('0') << t << "-before.avi";
        ssOutTxtPath << outDir << "/" << std::experimental::filesystem::path(videoPath).stem().generic_string() << "-" << patchType << "-" << std::setw(4) << std::setfill('0') << t << ".success";


        if (!std::experimental::filesystem::exists(outDir))
            std::experimental::filesystem::create_directory(outDir);

        int frameSize = 80;
        if (patchType == "mouth")
            frameSize = 80;

        std::vector<cv::Mat> frames = Aligner::readSeqClipFromVideo(videoPath, tBegin, tEnd, frameSize);

        if (!frames.size())
            break;

        std::vector<std::vector<double> > poses;
        std::vector<cv::Mat > landmarks;
        std::vector<cv::Mat> colorFullFrames;
        std::vector<cv::Mat>* ptrColorFullFrames = nullptr;

        if (patchType == "leye")
            ptrColorFullFrames = &colorFullFrames;

        std::vector<cv::Mat> outMagicIms;
        std::vector<Label> estLabels;
        std::vector<int> registrationStatus;

        std::vector<cv::Mat> estWarps;
        std::vector<bool> successFlag;

        //!std::vector<cv::Mat> subFrames(frames.begin(), frames.begin()+5);

        std::cout << "aligning sequence (" << patchType << "), part #" << t << std::endl;
        std::vector<cv::Mat> rframes = A.alignSequence(frames, outMagicIms, &estLabels, registrationStatus, nullptr, nullptr, &estWarps, &successFlag);
//        std::cout << "aligned sequence " << std::endl;

        A.saveRegisteredVideo(ss.str(), patchType, frames, outMagicIms,&estWarps,&registrationStatus, &poses, &landmarks, saveBefore, ssOutVidPath.str(), ssOutTxtPath.str(), FPS, ssBeforeVidPath.str());

        //! Save full faces too
        if (ptrColorFullFrames != nullptr)
        {
            for (size_t i=0; i<ptrColorFullFrames->size(); ++i)
            {
                std::stringstream ssi;
                ssi << ssFullFace.str() <<  "/" <<  std::setw(6) << std::setfill('0') << i+1 << ".png";

                cv::imwrite(ssi.str(), ptrColorFullFrames->at(i));
            }
        }
    }
}


















