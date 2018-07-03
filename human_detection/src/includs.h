#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>

// PCL specific includes
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>  
#include <pcl/filters/extract_indices.h> 
#include <pcl/filters/voxel_grid.h> 
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/ModelCoefficients.h>  
#include <pcl/octree/octree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>  
#include <pcl/segmentation/extract_clusters.h> 
#include <pcl/segmentation/sac_segmentation.h>

//ros <-> pcl
#include <pcl_conversions/pcl_conversions.h>

//Eigen include
//#include <Eigen/SVD>
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "Eigen/LU"
//#include <unsupported/Eigen/Splines>

//base include
#include <fstream>
#include <vector>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <time.h>

//other include
#include "svm_2.h"
#include <omp.h>
using namespace std;
using namespace Eigen;

