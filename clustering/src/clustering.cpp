#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Core>
#include <boost/thread.hpp>
using namespace std;
using namespace Eigen;

typedef pcl::PointXYZINormal PointA;
typedef pcl::PointCloud<PointA>  CloudA;
typedef pcl::PointCloud<PointA>::Ptr  CloudAPtr;

boost::mutex pt_mutex;
bool callback_flag=false;
CloudAPtr rmg_pt (new CloudA);

void getClusterInfo(CloudA pt, PointA & cluster)
{
    ////calculate_centroid_section
    // Vector3f centroid=Vector3f::Zero(3);
    Vector3f centroid;
	centroid[0]=pt.points[0].x;
	centroid[1]=pt.points[0].y;
	centroid[2]=pt.points[0].z;
    
	Vector2f min_p;
	min_p[0]=pt.points[0].x;
	min_p[1]=pt.points[0].y;
    
	Vector3f max_p;
	max_p[0]=pt.points[0].x;
	max_p[1]=pt.points[0].y;
	max_p[2]=pt.points[0].z;
	
	for(size_t i=1;i<pt.points.size();i++){
        centroid[0]+=pt.points[i].x;
        centroid[1]+=pt.points[i].y;
        centroid[2]+=pt.points[i].z;
		if (pt.points[i].x<min_p[0]) min_p[0]=pt.points[i].x;
		if (pt.points[i].y<min_p[1]) min_p[1]=pt.points[i].y;
		
		if (pt.points[i].x>max_p[0]) max_p[0]=pt.points[i].x;
		if (pt.points[i].y>max_p[1]) max_p[1]=pt.points[i].y;
		if (pt.points[i].z>max_p[2]) max_p[2]=pt.points[i].z;
    }
    
	cluster.x=centroid[0]/(float)pt.points.size();
    cluster.y=centroid[1]/(float)pt.points.size();
    cluster.z=centroid[2]/(float)pt.points.size();
	cluster.normal_x=max_p[0]-min_p[0];
	cluster.normal_y=max_p[1]-min_p[1];
	cluster.normal_z=max_p[2];
}

void cpu_clustering(CloudAPtr pcl_in, CloudA& cluster_centroid, CloudA& cluster_pt)
{
    //Downsample//
    pcl::VoxelGrid<pcl::PointXYZINormal> vg;  
	CloudAPtr ds_cloud (new CloudA);  
	vg.setInputCloud (pcl_in);  
	// vg.setLeafSize (0.001f, 0.001f, 0.001f);
	vg.setLeafSize (0.07f, 0.07f, 0.07f);
	// vg.setLeafSize (0.09f, 0.09f, 0.09f);
	vg.filter (*ds_cloud);
    
   
    //downsampled point's z =>0
    vector<float> tmp_z;
    tmp_z.resize(ds_cloud->points.size());
	for(int i=0;i<(int)ds_cloud->points.size();i++){
        tmp_z[i]=ds_cloud->points[i].z;
		ds_cloud->points[i].z  = 0.0;
    }
    
    
    //Clustering//
    // Creating the KdTree object for the search method of the extraction
    // clock_t clustering_start=clock();
    pcl::search::KdTree<PointA>::Ptr tree (new pcl::search::KdTree<PointA>);
    tree->setInputCloud (ds_cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZINormal> ec;
    ec.setClusterTolerance (0.15); // 15cm
    // ec.setClusterTolerance (0.50); // 15cm
    ec.setMinClusterSize (15);
    // ec.setMaxClusterSize (200);
    ec.setMaxClusterSize (800);
    ec.setSearchMethod (tree);
    ec.setInputCloud(ds_cloud);
    ec.extract (cluster_indices);
    // std::cout<<"clustering time="<<(double)(clock()-clustering_start)/CLOCKS_PER_SEC<<std::endl;
  
    //reset z value
	for(int i=0;i<(int)ds_cloud->points.size();i++)
        ds_cloud->points[i].z=tmp_z[i];
    
    //get cluster information//
    size_t num=0;
    cluster_centroid.resize(cluster_indices.size());
	for(size_t iii=0;iii<cluster_indices.size();iii++){
        //cluster cenrtroid
        CloudAPtr cloud_cluster (new CloudA);
		cloud_cluster->points.resize(cluster_indices[iii].indices.size());
        //cluster points
        cluster_pt.points.resize(cluster_indices[iii].indices.size()+num);
        for(size_t jjj=0;jjj<cluster_indices[iii].indices.size();jjj++){
            int p_num=cluster_indices[iii].indices[jjj];
            cloud_cluster->points[jjj]=ds_cloud->points[p_num];
            cluster_pt.points[num+jjj]=ds_cloud->points[p_num];
        }
        //get bounding box's centroid
        // Vector3f centroid=calculateCentroid(*cloud_cluster);
        getClusterInfo(*cloud_cluster, cluster_centroid[iii]);
        // cluster_centroid[iii].x=centroid[0];
        // cluster_centroid[iii].y=centroid[1];
        // cluster_centroid[iii].z=centroid[2];
        //the number of points which constitute cluster[iii]
        cluster_centroid[iii].curvature=cloud_cluster->points.size();
        //the start index of cluster[iii]
        // cluster_centroid[iii].normal_x=num;
        cluster_centroid[iii].intensity=num;
        num=cluster_pt.points.size();//save previous size
    }
    
}

void objectCallback(const sensor_msgs::PointCloud2::Ptr &msg)
{
	boost::mutex::scoped_lock(pt_mutex);
	pcl::fromROSMsg(*msg,*rmg_pt);
	callback_flag=true;
}

int main (int argc, char** argv)
{
	ros::init(argc, argv, "clustering");
	ros::NodeHandle nh,n;
  
    ros::Publisher cluster_centroid_pub=nh.advertise<sensor_msgs::PointCloud2>("/cluster/centroid",1);
    ros::Publisher cluster_points_pub=nh.advertise<sensor_msgs::PointCloud2>("/cluster/points",1);
	// ros::Subscriber object_sub=n.subscribe("/static_cloud", 1, objectCallback);
	ros::Subscriber object_sub=n.subscribe("/rm_ground2", 1, objectCallback);

    ros::Rate loop_rate(20);
    while (ros::ok()){
        if (callback_flag){
			// clock_t start=clock();
            CloudA cluster_centroid, cluster_points;
            cpu_clustering(rmg_pt, cluster_centroid, cluster_points);
            
            sensor_msgs::PointCloud2 cluster_centroid_ros;
            toROSMsg(cluster_centroid, cluster_centroid_ros);
            cluster_centroid_ros.header.frame_id="velodyne";
            cluster_centroid_ros.header.stamp=ros::Time::now();
            cluster_centroid_pub.publish(cluster_centroid_ros);
            
            sensor_msgs::PointCloud2 cluster_points_ros;
            toROSMsg(cluster_points, cluster_points_ros);
            cluster_points_ros.header.frame_id="velodyne";
            cluster_points_ros.header.stamp=ros::Time::now();
            cluster_points_pub.publish(cluster_points_ros);

			// cout<<(double)(clock()-start)/CLOCKS_PER_SEC<<endl<<endl;
            callback_flag=false;        
        } 
        ros::spinOnce();
        loop_rate.sleep();
    }

  return (0);
}
