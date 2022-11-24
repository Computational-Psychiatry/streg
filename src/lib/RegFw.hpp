#ifndef REGFW_HPP
#define REGFW_HPP
//#define VERB 1 // Verbose output -- see registration output etc/
//#define VERB_TIMING 1 // Verbose output -- see registration output etc/
//#define EXTRAVERB 1 // Verbose output -- see registration output etc/

#include <vector>
#include <string>
#include <map>
#include <initializer_list>

using std::map; using std::vector; using std::string;

namespace rfw
{
    const int tupleSize = 2;
    const int imW = 200;

    static std::map<std::string, int> imWs = {{"face",200},{"leye",50},{"reye",50},{"mouth",50}};

    const double RT = 8;
    const double EPS = 0.000000000000000000000000000000000000000001;
    const double LL_THRESHOLD = -550; // log likelihood threshold for validating registration

    const std::string TUPLES_PATH = "data/tuples/";
    const std::string TUPLES_FOR_THRESHOLD_PATH = "data/tuples_for_threshold/";
    const std::string TEST_TUPLES_FOR_THRESHOLD_PATH = "data/test/tuples_for_threshold/";
    const std::string TEST_ROCS_PATH = "data/test/failure_identification";
//    const std::string TUPLES_PATH = "data/rtuples/";
    const std::string SAMPLES_PATH_BASE = "data/samples/";
    const std::string TEST_SAMPLES_FOR_THRESHOLD_PATH = "data/test/samples_for_threshold/";
    const std::string CK_DBPATH = "ck/cohn-kanade-images_elim";
    const std::string CK_LPATH = rfw::CK_DBPATH+"/../Landmarks/";
    const std::string CMU_DBPATH = "cmu/sequences";
    const std::string CMU_LPATH = rfw::CMU_DBPATH+"/../Landmarks/";
    const std::string MMI_DBPATH = "mmi/sequences";
    const std::string MMI_LPATH = rfw::MMI_DBPATH+"/../Landmarks/";
    const std::string RESULTS_PATH = "data/results/";
    const std::string RESULTS_PAIRS_PATH = rfw::RESULTS_PATH+"/pairs/";
    const std::string RESULTS_SEQS_PATH = rfw::RESULTS_PATH+"/seqs/";
    const std::string VISUAL_PATH = "data/visual/";
    const std::string VISUAL_SEQS_PATH = rfw::VISUAL_PATH+ "/";
    const std::string VISUAL_PAIRS_PATH = rfw::VISUAL_PATH+ "/pairs/";
}


#endif // REGFW_HPP
