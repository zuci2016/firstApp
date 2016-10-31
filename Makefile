PROJECT := ssd_detect

INCLUDE_DIRS :=-I /home/zuci/zcLibs/cuda/include -I /usr/include -I /home/zuci/zcCaffe/caffe-SSD/caffe/include -I /usr/local/cuda/include

#INCLUDE_DIRS :=/home/zuci/zcCaffe/caffe-SSD/caffe/include


LIBRARY_DIRS := -L /usr/local/lib -L /usr/lib -L /home/zuci/zcCaffe/caffe-SSD/caffe/build/lib -L /home/zuci/zcLibs/cuda/lib64 -L /usr/local/cuda/lib64

OPENCV_LIB += -lopencv_core
OPENCV_LIB += -lopencv_highgui
#OPENCV_LIB += -lopencv_videoio

LIBRARIES := cudart cublas curand
LIBRARIES += glog gflags protobuf boost_system boost_filesystem hdf5
LIBRARIES += leveldb snappy 
LIBRARIES += lmdb
LIBRARIES += opencv_core opencv_highgui opencv_imgproc
LIBRARIES += boost_thread stdc++ cudnn
LIBRARIES += cblas atlas
LIBRARIES += caffe

LDLIB = $(foreach library,$(LIBRARIES),-l$(library))
#@echo LDLIB

all:build

build:ssd_detect
	#g++ -E ssd_detect.cpp -o ssd_detect.i -I $(INCLUDE_DIRS) 

ssd_detect.i:ssd_detect.cpp	
	g++ -E ssd_detect.cpp -o ssd_detect.i -I $(INCLUDE_DIRS)

ssd_detect.o:ssd_detect.i
	g++ -c ssd_detect.i -o ssd_detect.o

ssd_detect: ssd_detect.o
	$(warning $(LDLIB))
#	g++ ssd_detect.o -o ssd_detect $(LIBRARY_DIRS) $(OPENCV_LIB)
	g++ -o ssd_detect ssd_detect.o $(LIBRARY_DIRS) $(LDLIB)
	#g++ ssd_detect.o /home/zuci/zcCaffe/caffe-SSD/caffe/build/lib/libcaffe.a -o ssd_detect
