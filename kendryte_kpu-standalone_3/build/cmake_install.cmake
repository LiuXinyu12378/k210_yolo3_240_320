# Install script for directory: C:/Users/Administrator/Desktop/K210_Yolo_framework/K210_Yolo_framework/kendryte_kpu-standalone_3

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/kendryte_kpu-standalone")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/Administrator/Desktop/K210_Yolo_framework/K210_Yolo_framework/kendryte_kpu-standalone_3/build/kendryte_standalone-sdk/cmake_install.cmake")
  include("C:/Users/Administrator/Desktop/K210_Yolo_framework/K210_Yolo_framework/kendryte_kpu-standalone_3/build/kendryte_camera-standalone-driver/cmake_install.cmake")
  include("C:/Users/Administrator/Desktop/K210_Yolo_framework/K210_Yolo_framework/kendryte_kpu-standalone_3/build/kendryte_lcd-nt35310-standalone-driver/cmake_install.cmake")
  include("C:/Users/Administrator/Desktop/K210_Yolo_framework/K210_Yolo_framework/kendryte_kpu-standalone_3/build/kendryte_w25qxx-standalone-driver/cmake_install.cmake")
  include("C:/Users/Administrator/Desktop/K210_Yolo_framework/K210_Yolo_framework/kendryte_kpu-standalone_3/build/kendryte_ai_image/cmake_install.cmake")

endif()

