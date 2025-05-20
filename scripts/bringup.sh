#!/bin/bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

ros2 launch xarm_moveit_servo xarm_moveit_servo_realmove.launch.py \
    dof:=7 robot_ip:=192.168.1.231 add_gripper:=true report_type:=dev
