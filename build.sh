CUDA_HOME=/usr/local/cuda
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


#cp ./lib/cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/

nvcc -std=c++11 -ccbin=/usr/bin/g++-4.9 -c -o lib/active_shift2d.cu.o lib/active_shift2d.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
-I $TF_INC \
-L /usr/local/cuda-8.0/lib64/ -I /usr/local/ -I $TF_INC/external/nsync/public \
--expt-relaxed-constexpr -gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50

g++-4.9 -std=c++11 -shared -o lib/active_shift2d.so lib/active_shift2d.cc lib/active_shift2d.cu.o \
-I $TF_INC -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors \
-I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -ltensorflow_framework -I $TF_INC/external/nsync/public
