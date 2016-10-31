// This is a demo code for using a SSD model to do detection.
//
//
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
//
#define USE_OPENCV

#include "/home/zuci/zcCaffe/caffe-SSD/caffe/include/caffe/caffe.hpp"
#include <boost/filesystem.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <map>
#include <dirent.h>
#include <stdlib.h>
#include <math.h>
//#include <iostream>
#include <fstream>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace boost::filesystem;
using namespace std;

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

class Identifier{
	public:
		Identifier(const string& model_file,const string& weights_file,const string& mean_file);
		
		//id_floder is the floder that hold the person need to be found,newWidth and newHeight will resize the input imgs to new size for identify
		//it will create the interested person's features
		bool Init(const string& id_floder,const string& blob_names);
		
		//do now!
		//use NN to classify the input image!the predicted id returned by predicted_id,and the NN distance returned by distNN
		bool Identify(const cv::Mat& inputImg,int* predicted_id,double* distNN);
		
		//Extract features from given blob's names,the data is hold in shared_ptr:features_ptr
		void ExactFeatures(const cv::Mat& inputMat,std::vector<std::string>& vect_blob_names,shared_ptr<float>& features_ptr);
 
	private:
		//Read file list from given dir to vector of file names
		void ReadFileList(const string& id_floder,std::vector<std::string>& fileNames);
		
		void SetMean(const string& mean_file);
		void WarpInputLayer(std::vector<cv::Mat>* input_channels);
		void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
		void ComputeNN(shared_ptr<float>& data_ptr1,shared_ptr<float>& data_ptr2,double* distNN);
		

	private:
		std::vector<int > vect_interested_Ids_;                    //the ids that needed to found
		bool binit_;                                              //the flag that judge is init first,evey usr need init first!
		std::vector<shared_ptr<float> > vect_interested_features_; //vector that hold the interested images's features pairs 
		long features_dims_;
		std::vector<std::string > vect_blob_names_;                //vector that hold the will be extracted blobs's names

		shared_ptr<Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		cv::Mat mean_;
};

Identifier::Identifier(const string& model_file,const string& weights_file,const string& mean_file)
{
	binit_ = false;
	Caffe::set_mode(Caffe::GPU);
	//Load caffe net and it's weights
	net_.reset(new Net<float>(model_file,TEST));
	net_->CopyTrainedLayersFrom(weights_file);

	CHECK_EQ(net_->num_inputs(),1)<<"Netwrok should have exactly one input.";
	CHECK_EQ(net_->num_outputs(),1)<<"Network should have exactly one output.";
	
	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_==3 || num_channels_ == 1)
			<<"Input channels should be 1 or 3 channels";
	input_geometry_ = cv::Size(input_layer->width(),input_layer->height());

	//load mean file
	SetMean(mean_file);
}

void Identifier::ComputeNN(shared_ptr<float>& data_ptr1,shared_ptr<float>& data_ptr2,double* distNN)
{
	double NN = 0.0;
	float* data1 = data_ptr1.get();
	float* data2 = data_ptr2.get();
	for(int i=0;i<features_dims_;++i)
	{
		NN += std::abs(data1[i] - data2[i]);
	}	
	*distNN = NN;	
}

bool Identifier::Init(const string& id_floder,const std::string& cst_blob_names)
{
	//read data form in_dloder
	CHECK(!cst_blob_names.empty())<<"You should specify which blobs need to extract features!It cannot be empty!";
	std::vector<std::string> vect_paths;
    ReadFileList(id_floder,vect_paths);
	CHECK(!vect_paths.empty())<<"No interested person's img is imputed!";
	if(!vect_blob_names_.empty())
		vect_blob_names_.clear();
	//erase first and last of " "
	std::string blob_names(cst_blob_names);
	blob_names.erase(0,blob_names.find_first_not_of(" "));
	blob_names.erase(blob_names.find_last_not_of(" ")+1);
	//split the blob_names string to vector of blob name that need to extracted to shape the features
	//std::vector<std::string> vect_blobNames;
	size_t last = 0;
	size_t index = blob_names.find_first_of(" ");
	if(index == std::string::npos)
	{
		vect_blob_names_.push_back(blob_names);
	}
	else
	{
		std::string downSampleStr = blob_names;			
		while(index!= std::string::npos)
		{
			//std::string downSampleStr = blob_names;
			//std::string current_blobName = blob_names.substr(last,index -last);
			vect_blob_names_.push_back(downSampleStr.substr(0,index));
			//the case is for handling blob_names like "fc7  fc8",between "fc7" and "fc8",there has two " "
			downSampleStr = blob_names.substr(index+1);
			if(downSampleStr[0] == ' ')
				downSampleStr.erase(0,downSampleStr.find_first_not_of(" "));
			index = downSampleStr.find_first_of(" ");
		}
	}

	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1,num_channels_,input_geometry_.height,input_geometry_.width);
	net_->Reshape();
	
	//Extract features from interested preson
	for(int i=0;i<vect_paths.size();++i)
	{
		cv::Mat personMat = cv::imread(vect_paths[i]); //read image 
		shared_ptr<float> current_ptr;
		ExactFeatures(personMat,vect_blob_names_,current_ptr);
		vect_interested_features_.push_back(current_ptr);
	}
	binit_ = true;
	return true;
}

void Identifier::ReadFileList(const std::string& id_floder,std::vector<std::string>& fileNames)
{
	CHECK(!id_floder.empty())<<"The interested person's image's dir cannot be empty!";
	if(!fileNames.empty())
		fileNames.clear();
	
	boost::filesystem::path dirP(id_floder.c_str());
	CHECK(boost::filesystem::exists(dirP))<<"The interested person's images's dir is not exit!";
	DIR* dir;
	struct dirent *ptr_dir;

	//open dir
	CHECK(NULL != (dir = opendir(id_floder.c_str())))<<"Cannot open the dir to read the interested person!";
	
	while((ptr_dir = readdir(dir)) != NULL)
	{
		if(strcmp(ptr_dir->d_name,".") == 0 || strcmp(ptr_dir->d_name,"..") == 0)
			continue;
		if(8 == ptr_dir->d_type)//it means it is a file type!
		{
			std::string strFileName = std::string(ptr_dir->d_name);
			int currentId = atoi(strFileName.c_str())/100;
			vect_interested_Ids_.push_back(currentId);
			std::string fullStr = id_floder + "/" + strFileName;
			boost::filesystem::path currentP(fullStr.c_str());
			CHECK(boost::filesystem::exists(currentP))<<"Cannot find the image!"<<currentP.string();
			fileNames.push_back(currentP.string());
		}
	}
}

void Identifier::ExactFeatures(const cv::Mat& inputImg,std::vector<std::string>& vect_blob_names,shared_ptr<float>& features_ptr)
{
	std::vector<cv::Mat> input_channels;
	WarpInputLayer(&input_channels);
    
	Preprocess(inputImg, &input_channels);
    net_->Forward();
	
	size_t num_blobs = vect_blob_names.size();

	for(size_t i =0;i<num_blobs;++i)
		CHECK(net_->has_blob(vect_blob_names[i]))<<"Unkown blob name"<<vect_blob_names[i]<< "in the caffe net!";

	//compute features dims length
	int total_features_dims = 0;
	for(int i=0;i<num_blobs;++i)
	{
		const boost::shared_ptr<Blob<float> > feature_blob = net_->blob_by_name(vect_blob_names[i]);
		total_features_dims += feature_blob->count();
	}
	features_ptr = shared_ptr<float>(new float[total_features_dims]);
	features_dims_ = total_features_dims;
	
	//copy data to shared_ptr
	int offSet = 0;
	for (int i=0;i<num_blobs;++i)
	{
		const boost::shared_ptr<Blob<float> > feature_blob = net_->blob_by_name(vect_blob_names[i]);
		memcpy(features_ptr.get()+offSet,feature_blob->cpu_data(),sizeof(float)*feature_blob->count());
		offSet += feature_blob->count();
	}
}

bool Identifier::Identify(const cv::Mat& inputImg,int* predicted_id,double* distNN)
{
	CHECK(binit_)<<"Need init before indentify,please command function Init first!";
	shared_ptr<float> data_test;
	ExactFeatures(inputImg,vect_blob_names_,data_test);
	
	//computes distNNs and get min NN distance with it's IDs
	double minNN =0.0;
    ComputeNN(data_test,vect_interested_features_[0],&minNN);	
	int ids = vect_interested_Ids_[0];
	for(int i=0;i<vect_interested_Ids_.size();++i)
	{
		double currentNN = 0.0;
		ComputeNN(data_test,vect_interested_features_[i],&currentNN);
		if(minNN > currentNN)
		{
			minNN = currentNN;
			ids = vect_interested_Ids_[i];
		}
	}	
	//return the result
	* predicted_id = ids;
	* distNN = minNN;
	return true;
}

/* Load the mean file in binaryproto format. */
void Identifier::SetMean(const string& mean_file) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    //CHECK(mean_value.empty()) <<
      //"Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Identifier::WarpInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Identifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */

  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
  {
    cv::resize(sample, sample_resized, input_geometry_);
  }
  else
  {
    sample_resized = sample;
  }

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;

  //WARNNING:---------------------------------------------------------------------------------------
  //unment in 2016/10/26 because I didnot use mean file
  //cv::subtract(sample_float, mean_, sample_normalized);
	sample_normalized = sample_float.clone();
  //------------------------------------------------------------------------------------------------

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

//  printf("-----------------comehere!?");
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
 // Caffe::set_mode(Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  
		//-------------TO DO!--------------------------
  //---------zuci 2016/10/27 say:it is not aways to reshape the whole net,if your input image size is always same,so you can revise these codes------------
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  //--------------------------not always reshape revise  TO DO!----------------------------------------------
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
  {
    cv::resize(sample, sample_resized, input_geometry_);
  }
  else
  {
    sample_resized = sample;
  }

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;

  //WARNNING:---------------------------------------------------------------------------------------
  //unment in 2016/10/26 because I didnot use mean file
  //cv::subtract(sample_float, mean_, sample_normalized);
	sample_normalized = sample_float.clone();

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

//  printf("-----------------comehere!?");
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");


int main(int argc, char** argv) {
		/*
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }
  */


   string model_file = "";
   string weights_file = "";
   string mean_file = "";
   string mean_value = "";//"104,117,123";
   string video_file = "";//string(argv[3]);
 // const string& file_type = FLAGS_file_type;
   string out_file = "./output_file.html";
   float confidence_threshold = 0.15;
	
  //indentifier's params
   string id_model_file = "";//string(argv[4]);
   string id_weight_file = "";//string(argv[5]);
   string id_mean_file = "";//string(argv[7]);
   string id_train_file = "";//string(argv[6]);
   string id_blob_names = "";//string(argv[7]);
   float yuzhi = 0.0;

  const string& strParamsTxt = string(argv[1]);

  //std::string line;
  std::ifstream infile(strParamsTxt.c_str());
  if(!infile)
  {
	printf("Cannot open the params txt!");
	return -1;
  }

  getline(infile,model_file);
  getline(infile,weights_file);
  getline(infile,video_file);
  getline(infile,id_model_file);
  getline(infile,id_weight_file);
  getline(infile,id_train_file);
  getline(infile,id_blob_names);
  std::string yuzhi_str = "500.0";
  getline(infile,yuzhi_str);
  yuzhi = atof(yuzhi_str.c_str());
  



  //Init the identify network
  Identifier identifier(id_model_file,id_weight_file,id_mean_file);

  printf("Identifier is initting ... ...");

  identifier.Init(id_train_file,id_blob_names);

  printf("Identifier is init coplete!");

  // Initialize the network.
  Detector detector(model_file, weights_file, mean_file, mean_value);

  // Process image one by one.
  //
  //----------------------unmented for debug!-------------------------------------
  cv::VideoCapture cap(video_file);
  if (!cap.isOpened()) {
        LOG(FATAL) << "Failed to open video: " << video_file;
      }
      cv::Mat img;
      int frame_count = 0;
      while (true) {
        bool success = cap.read(img);
        if (!success) {
          LOG(INFO) << "Process " << frame_count << " frames from " << video_file;
          break;
        }
        CHECK(!img.empty()) << "Error when read frame";
		//resize img
		cv::resize(img,img,cv::Size(img.cols/2,img.rows/2));
        std::vector<vector<float> > detections = detector.Detect(img);
        // Print the detection results. 
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= confidence_threshold) {
			if(static_cast<int>(d[1])==15)
			{
				int x1 = static_cast<int>(d[3]*img.cols);
				int y1 = static_cast<int>(d[4]*img.rows);
				int x2 = static_cast<int>(d[5]*img.cols);
				int y2 = static_cast<int>(d[6]*img.rows);
				cv::Mat personMat = img(cv::Rect(x1,y1,x2-x1,y2-y1));
				double distNN = 0.0;
				int person_id = -1;
				identifier.Identify(personMat,&person_id,&distNN);
				rectangle(img,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,0,255),2.0);
				//give a yuzhi to judge a person is a interesting person!
				if(distNN < yuzhi)
				{
					//put text to the img
					char textStr_NN[10];
					std::sprintf(textStr_NN,"%f",distNN);
					char textStr_id[5];
					std::sprintf(textStr_id,"%d",person_id);
					std::string scrStr = string(textStr_NN) +" ID:" + string(textStr_id);
					putText(img,scrStr,cv::Point(x1,y1),CV_FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,255,0));
					//--------------save the image! Release version could comment it to get faster speed!---------------------
					std::string strSavePath =  "/home/zuci/zcXM/detectAvi/model/test_craime/" + scrStr + ".jpg";
					imwrite(strSavePath,personMat);
					//---------------The Save code is also too simple to transplant it to other plantforms---------------------
				}
			}
          }
        }
		cv::imshow("test",img);
		cvWaitKey(1);
        ++frame_count;

		//----debug----------------
		//-------------------------
      }
      if (cap.isOpened()) {
        cap.release();
      }
  //--------------------------------------------------------------
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
