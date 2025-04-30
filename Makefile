# Compiler and flags
# 定义编译器
CC = nvcc

# 定义编译选项，这里可以根据需要添加优化选项等
# CCFLAGS = -O3 -march=native -funroll-loops -ffast-math -flto -Wunused-result -Wno-return-type -fprefetch-loop-arrays -I./src -I./include -I/home/xinxiangsama/lib/hdf5-1.14.6/include -DH5_BUILD_AS_DYNAMIC_LIB
CCFLAGS = -O3  -Wno-deprecated-gpu-targets -I/usr/local/cuda-12.8/include -I./src -I./include -I/usr/local/include/vtk-9.4 -I/home/xinxiangsama/lib/hdf5-1.14.6/include -DH5_BUILD_AS_DYNAMIC_LIB 

# 定义链接选项
# LDFLAGS = -lboost_system -lboost_filesystem -L/home/xinxiangsama/lib/hdf5-1.14.6/lib -lhdf5_cpp -lhdf5
LDFLAGS = -L/usr/local/lib \
          -lvtkCommonCore-9.4 \
          -lvtkCommonDataModel-9.4 \
          -lvtkIOXML-9.4 \
			-lvtkIOParallelXML-9.4 \
          -lvtkRenderingOpenGL2-9.4 \
          -lvtkRenderingCore-9.4 \
          -lvtksys-9.4 \
		  -lvtkParallelCore-9.4\
          -L/home/xinxiangsama/lib/hdf5-1.14.6/lib \
          -lhdf5_cpp -lhdf5


# 定义源文件
SOURCES = \
	./src/main.cpp \
	./src/Run.cpp \
	./src/particle/Particle.cpp ./src/particle/GPUParticle.cu\
	./src/cell/Cell.cpp ./src/cell/GPUCell.cu\
	./src/meshes/Element.cpp  ./src/meshes/CartesianMesh.cpp \
	./src/random/Random.cpp \
	./src/boundary/WallBoundary.cpp ./src/boundary/OutletBoundary.cpp ./src/boundary/PeriodicBoundary.cpp ./src/boundary/InletBoundary.cpp\
	./src/io/Output.cpp \
	./src/object/Segment.cpp ./src/object/Geom.cpp ./src/object/Circle.cpp ./src/object/Square.cpp \
	

# 定义目标文件
OBJECTS = $(patsubst %.cpp, build/%.o, $(SOURCES))

# 定义最终的可执行文件名
EXECUTABLE = runsim

# 默认目标
all: $(EXECUTABLE)

# 链接目标文件生成可执行文件
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

# 从.cpp文件编译出.o文件
build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@

# 清理规则
clean:
# rm -f $(OBJECTS) $(EXECUTABLE)
	rm -rf ./build/src
	rm -rf ./runsim

# 清理res文件夹下的.h5文件
clean_result:
	rm -rf ./res/*.h5
	rm -rf ./res/*.vts
	rm -rf ./res/*.vtu
	rm -rf ./res/*.pvts
	rm -rf ./res/*

run :
	mpiexec -np 4 --oversubscribe ./runsim
# 防止make自动生成*.o文件的规则
.PHONY: all clean clean_result
