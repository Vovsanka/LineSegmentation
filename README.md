# LineSegmentation

On Linux: g++, cmake, opencv (with CUDA support), cairo
libgtk2.0-dev pkg-config,
libcanberra-gtk-module libcanberra-gtk3-module,



Build opencv with:

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D BUILD_opencv_cudacodec=OFF \
      -D WITH_GTK=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      ..

make -j$(nproc)
sudo make install