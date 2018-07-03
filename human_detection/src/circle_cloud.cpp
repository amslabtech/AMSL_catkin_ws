#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;

sensor_msgs::PointCloud circles;
int main(int argc, char** argv)
{
	ros::init(argc, argv, "circle_cloud");
	ros::NodeHandle n;

	ros::Publisher pub = n.advertise<sensor_msgs::PointCloud>("/circle_cloud", 100);
	ros::Rate loop_rate(1);

	float r=0;
	cout<<"r= ";
	cin >> r; 
	while (ros::ok()) {
		float theta=0;

		for(int i=0;i<1000;i++){
			geometry_msgs::Point32 circle;

			//r=10.0;
			//theta=M_PI/4.0;
			theta=((2*M_PI)/1000)*i;		

			circle.x=r*cos(theta);
			circle.y=r*sin(theta);
			circle.z=-1.35;
			// circle.z=0;//-2.0;
			circles.points.push_back(circle);	
		}		


		circles.header.frame_id="/velodyne";
		circles.header.stamp=ros::Time::now();
		pub.publish(circles);

		circles.points.clear();


		ros::spinOnce();
		loop_rate.sleep();
	}
	ros::spin();
}

