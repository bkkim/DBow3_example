

#include <iostream>
#include <vector>
#include <filesystem>


// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>

#include "DescManip.h"


using namespace DBoW3;
using namespace std;
using std::filesystem::directory_iterator;  // c++ 17



//command line parser
class CmdLineParser {
	int argc;
	char** argv;
public:
	CmdLineParser(int _argc, char** _argv)
		: argc(_argc)
		, argv(_argv)
	{}

	bool operator[] (string param) {
		int idx = -1;
		for (int i = 0; i < argc && idx == -1; i++)
			if (string(argv[1]) == param)
				idx = i;
		return(idx != -1);
	}

	string operator()(string param, string defvalue = "-1") {
		int idx = -1;
		for (int i = 0; i < argc && idx == -1; i++)
			if (string(argv[i]) == param)
				idx = i;
		if (idx == -1)
			return defvalue;
		else
			return (argv[idx + 1]);
	}
};


vector<string> readImagePath(string path) 
{
	vector<string> images;
	for (const auto& file : directory_iterator(path))
	{
		images.push_back(path+"/"+file.path().filename().string());
	}

	return images;
}

vector<cv::Mat> loadOrbFeatures(std::vector<string> images, string descriptor = "") throw(std::exception) {
	// select detector
	cv::Ptr<cv::Feature2D> fdetector;
	if (descriptor == "orb")
		fdetector = cv::ORB::create();
	else throw std::runtime_error("Invalid descriptor");

	assert(!descriptor.empty());
	vector<cv::Mat> features;

	cout << "Extracting   features..." << endl;
	for (size_t i = 0; i < images.size(); i++)
	{
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		cout << "reading image: " << images[i] << endl;
		cv::Mat image = cv::imread(images[i], 0);
		if (image.empty())
			throw std::runtime_error("Could not open image " + images[i]);
		cout << "extracting features" << endl;
		fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
		features.push_back(descriptors);
		cout << "done detecting features" << endl;
	}

	return features;
}


void createVoc(const vector<cv::Mat>& features)
{
	// branching factor and depth levels
	const int k = 9;
	const int L = 3;
	const WeightingType weight = TF_IDF;
	const ScoringType score = L1_NORM;

	DBoW3::Vocabulary voc(k, L, weight, score);

	cout << "Create a small " << k << "^" << L << " vocabulary..." << endl;
	voc.create(features);
	cout << "... done!" << endl;

	cout << "Vocabulary information: " << endl
		<< voc << endl << endl;

	// lets do something with this vocabulary
//	cout << "Matching images against themselves (0 low, 1 high): " << endl;
//	BowVector v1, v2;
//	for (size_t i = 0; i < features.size(); i++)
//	{
//		voc.transform(features[i], v1);
//		for (size_t j = 0; j < features.size(); j++)
//		{
//			voc.transform(features[j], v2);
//
//			double score = voc.score(v1, v2);
//			cout << "Image " << i << " vs Image " << j << ": " << score << endl;
//		}
//	}

	// save the vocabulary to disk
	cout << endl << "Saving vocabulary..." << endl;
	voc.save("small_voc.yml.gz");
	cout << "Done" << endl;
}


void testVoc(const vector<cv::Mat>& features_db, const vector<cv::Mat>& features_test, const vector<string>& images_db, const vector<string>& images_test)
{
	cout << "Creating a small database..." << endl;

	// load the vocabulary from disk
	Vocabulary voc("small_voc.yml.gz");

	Database db(voc, false, 0); // false = do not use direct index
	// (so ignore the last param)
	// The direct index is useful if we wnat to retrieve the features that
	// belong to some vocabulary node.
	// db creates a copy of the vocabulary, we may get rid of "voc" now
	
	// add images to the database
	for (size_t i = 0; i < features_db.size(); i++)
		db.add(features_db[i]);

	cout << "... done!" << endl;

	cout << "Database information: " << endl << db << endl;

	// and query the database
	cout << "Querying the test_set: " << endl;

	QueryResults results;
	for (size_t i = 0; i < features_test.size(); i++)
	{
		db.query(features_test[i], results, 4);
		cout << "Searching for Image " << i << ". " << results << endl;

		if (1/*DEBUG_IMAGE*/)
		{
			cv::Mat img_test = cv::imread(images_test[i], 0);
			cv::putText(img_test, "File: " + images_test[i], cv::Point2f(5, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 2);
			for (size_t i = 0; i < results.size(); i++)
			{
				// Image load
				cv::Mat img_db = cv::imread(images_db[results[i].Id], 0);
				std::string caption = "Number: " + to_string(i) + ", Index: " + to_string(results[i].Id) + ", Score: " + to_string(results[i].Score);
				cv::putText(img_db, "File: " + images_db[results[i].Id], cv::Point2f(5, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 2);
				cv::putText(img_db, caption, cv::Point2f(5, 40), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 2);
				cv::hconcat(img_db, img_test, img_db);
				cv::imshow("Search image", img_db);
				cv::waitKey(0);
			}
		}
	}

	cout << endl;

	// we can save the database.
	cout << "Saving database..." << endl;
	db.save("small_db.yml.gz");
	cout << "... done!" << endl;

	// once saved, we can load it again
	cout << "Retrieving database once again..." << endl;
	Database db2("small_db.yml.gz");
	cout << "... done! This is: " << endl << db2 << endl;
}


int main(int argc, char** argv)
{
	try {
		CmdLineParser cml(argc, argv);
		if (cml["-h"] || argc <= 2)
		{
			cerr << "Usage: descriptor_name  path_db  path_test\n\t descriptors:brisk,surf,orb,akaze(only if using opencv3)" << endl;
			return -1;
		}

		string descriptor = argv[1];

		auto images_db = readImagePath(string(argv[2]));
		vector<cv::Mat> features_db = loadOrbFeatures(images_db, descriptor);
		createVoc(features_db);

		auto images_test = readImagePath(string(argv[3]));
		vector<cv::Mat> features_test = loadOrbFeatures(images_test, descriptor);
		testVoc(features_db, features_test, images_db, images_test);
	}
	catch (std::exception& e) {
		cerr << e.what() << endl;
	}

	return 0;
}
