cmake_minimum_required(VERSION 3.0.2)
project(audio_play)

find_package(
  catkin REQUIRED COMPONENTS
  roscpp
  rospy
  audio_common_msgs
)

find_package(PkgConfig)
pkg_check_modules(GST1.0 gstreamer-1.0 REQUIRED)

find_package(Boost REQUIRED COMPONENTS thread)

include_directories(${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${GST1.0_INCLUDE_DIRS})

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  DEPENDS Boost GST1.0
  LIBRARIES audio_play
)

add_executable(
  audio_play
  src/audio_play.cpp
)

add_dependencies(
  audio_play
  ${${audio_capture}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(
  audio_play
  ${catkin_LIBRARIES}
  ${GST1.0_LIBRARIES}
  ${Boost_LIBRARIES}
)


install(
  TARGETS audio_play
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
