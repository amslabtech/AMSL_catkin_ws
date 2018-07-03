<?xml version="1.0"?>
<launch>
  <node name="velodyne_index_2time" pkg="classification" type="velodyne_index_2time" />
  <node name="rm_node20" pkg="velodyne_height_map" type="rm_node20" />
  <node name="cpu_clustering" pkg="clustering" type="cpu_clustering" />
  <node name="test_human_detection" pkg="classification" type="test_human_detection_v2" output="screen" />
  <node name="test_kf" pkg="tracking" type="test_kf"/>
  
  <node name="index_objects" pkg="classification" type="index_objects"/>
  <node name="rm_ground_node_minmax" pkg="velodyne_height_map" type="rm_ground_node_minmax"/>
</launch>
