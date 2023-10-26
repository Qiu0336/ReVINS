echo "Configuring and building DBow3 ..."

cd DBow3
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../..

echo "Configuring and building ReVINS ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

