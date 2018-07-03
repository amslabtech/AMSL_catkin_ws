#include "includs.h"
#define SAME_DIST 0.26

bool callback_flag=false;

pcl::PointCloud<pcl::PointNormal> h_pos; 

inline float calcDist(pcl::PointXYZI& p1, pcl::PointXYZI& p2)
{
	return hypot(fabs(p1.x-p2.x), fabs(p1.y-p2.y));
}

void HumanPositionCallback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	pcl::fromROSMsg(*msgs, h_pos);
	callback_flag = true;
}

int main(int argc, char** argv)
{

	ros::init(argc, argv, "visualize_human_id");
	ros::NodeHandle n;

	ros::Subscriber human_pos_sub= n.subscribe("/human_recognition/positive_position", 1, HumanPositionCallback);
	ros::Publisher human_id_pub = n.advertise<visualization_msgs::MarkerArray>("/human_recognition/human_id",1);

	pcl::PointCloud<PointNormal> pre_list; 
	pcl::PointCloud<PointNormal> now_list; 
	int id=0;
	
	ros::Rate loop_rate(20);
	while (ros::ok()){
		if (callback_flag){
			if (now_list.points.size()){
				//check the same people//
				bool add_flag=true;
				for (unsigned int i=0; i<h_pos.points.size(); ++i){
					for (unsigned int j=0; j<now_list.points.size(); ++j){
						if (SAME_DIST>calcDIST(h_pos.points[i], now_list.points[j])){
							add_flag=false;
							break;
						}
					}
					if (add_flag){
						h_pos.points[i].curvature= id++;
						now_list.push_back(h_pos.points[i]);
					} 
					else add_flag=true;
				}
				//check the disappeared people//
				bool erase_flag=true;
				for (unsigned int i=0; i<pre_list.points.size(); ++i){
					for (unsigned int j=0; j<now_list.points.size(); ++j){
						if (pre_list.points[i].curvature==now_list.points[j].curvature){
							erase_flag=false;
							break;
						}
					}
					if (erase_flag)
						now_list.erase(std::remove(now_list.begin(), now_list.begin()+i), now_list.end());
					else erase_flag=true;
				}
			}
			else{
				now_list.points.resize(h_pos.points.size());
				for (unsigned int i=0; i<h_pos.points.size(); ++i){
					now_list.points[i].x=h_pos.points[i].x;
					now_list.points[i].y=h_pos.points[i].y;
					now_list.points[i].z=h_pos.points[i].z;
					now_list.points[i].curvature= id++;
				}
			}

			//publish msg//
			// visualization_msgs::MarkerArray h_id_all;
			// visualization_msgs::Marker h_id;
			// h_id.header.frame_id="/velodyne";
			// h_id.header.stamp=ros::Time::now();
			// h_id.ns="test";
			// h_id.action=visualization_msgs::Marker::ADD;
			// h_id.pose.orientation.w=1.0;
			// h_id.type=visualization_msgs::Marker::TEXT_VIEW_FACING;
			// h_id.scale.x=0.5;
			// h_id.scale.y=0.5;
			// h_id.scale.z=0.5;
			// h_id.color.r=0.0;
			// h_id.color.g=1.0;
			// h_id.color.b=1.0;
			// h_id.color.a=1.0;
	
			// for(size_t i=0;i<now_list.size();i++){
				// h_id.id=i;
				// h_id.pose.position.x=now_list.points[i].x;
				// h_id.pose.position.y=now_list.points[i].y;
				// h_id.pose.position.z=now_list.points[i].z+1.0;
				
				// float val=now_list.points[i].curvature;
				// stringstream ss;
				// ss<<setprecision(3)<<val;
				// string ts=ss.str(); 
				// h_id.text= ts;
				// h_id.lifetime=ros::Duration(0.07);
				// h_id_all.markers.push_back(h_id);
				
			// }
			// human_id_pub.publish(h_id_all);

			//save now_list//
			pre_list.points.resize(now_list.points.size());
			pre_list=now_list;
		}
		ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}
