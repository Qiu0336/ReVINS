

ReVINS
===============

## Description

We have revisited monocular VIO and investigated feature tracking and IMU preintegration algorithms in this article. We proposed a robust IMU-aided feature tracker, in which two tracking strategies are utilized for depth-known and depth-unknown cases respectively. Then IMU preintegration and noise propagation on SE2(3) group is introduced, which indicates a more accurate uncertainty of integration terms. Furthermore, the presented feature tracker and IMU equations are both integrated in our newly designed three thread VIO system, including preprocessing, optimization and loop closing. Sliding-window bundle adjustment is perform on SE2(3). In addition, a decoupled loop closing module is presented in the system, which can be implemented in arbitrary VOor VIO systems.

**Paper:** Junyin Qiu, Weibo Huang, Hong Liu, Tianwei Zhang. **Revisiting Feature Tracking and IMU Preintegration for Monocular Visual-Inertial Odometry**.

# Requirement
## Boost
## Eigen3
## OpenCV 3.4
## Pangolin >= 0.5
## Ceres >= 2.0.0


# Build
Clone the repository:

```
git clone https://github.com/Qiu0336/ReVINS.git ReVINS
```

```
cd ReVINS
./build.sh
```

# Run

The config file indicates the path of the sequence, including ImagePath, TimestampPath and ImuPath.
ImagePath and ImuPath are paths of the original dateset such as EuRoC and TUMVI, while the TimestampPath is listed in the file config_example.
After setting in config file, run:

```
./build/ReVINS
```

Results are saved in the file /result when finishing the sequence.

## License

ReVINS is under [GPLv3 license](https://github.com/Qiu0336/ReVINS/blob/main/LICENSE).


