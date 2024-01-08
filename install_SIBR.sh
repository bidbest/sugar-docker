echo "This scribt will install sibr viewrs. PLease run this script within the docker"

cd /sugar/submodules/gaussian-splatting-docker/SIBR_viewers/cmake/linux
sed -i 's/find_package(OpenCV 4\.5 REQUIRED)/find_package(OpenCV 4.2 REQUIRED)/g' dependencies.cmake
sed -i 's/find_package(embree 3\.0 )/find_package(EMBREE)/g' dependencies.cmake
mv /sugar/submodules/gaussian-splatting-docker/SIBR_viewers/cmake/linux/Modules/FindEmbree.cmake /sugar/submodules/gaussian-splatting-docker/SIBR_viewers/cmake/linux/Modules/FindEMBREE.cmake
sed -i 's/\bembree\b/embree3/g' /sugar/submodules/gaussian-splatting-docker/SIBR_viewers/src/core/raycaster/CMakeLists.txt