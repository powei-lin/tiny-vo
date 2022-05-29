# tiny-vo

This project shows how minimun visual odometry works. Currently the process does not contain bundle adjustment.

<a href="https://youtu.be/NBjQhl4TGsU" target="_blank"><img src="/image/image.png" 
alt="tiny-vo" width="540" /></a>

## Build
```sh
git clone --recursive https://github.com/powei-lin/tiny-vo.git
cd tiny-vo && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Resease ..
make
```

## Download TUM-VI dataset
```sh
# https://vision.in.tum.de/data/datasets/visual-inertial-dataset
wget https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-corridor4_512_16.tar
```

## Run
```sh
./main -i dataset-corridor4_512_16 -c config/eucm_512.json -d config/tum_vi_dataset.json
```

## TODO (Probably😪)
* [ ] IMU preintegration
* [ ] Bundle Adjustment
* [ ] KeyFrame
