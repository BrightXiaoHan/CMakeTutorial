# 1、准备工作，编译方式、目标文件名、依赖库路径的定义。
CC = g++
CFLAGS  := -Wall -O3 -std=c++11 

OBJS = libadd.o #.o文件与.cpp文件同名
LIB = libadd.so # 目标文件名
INCLUDE = ./ # 头文件目录
HEADER = libadd.h

all : $(LIB)

# 2. 生成.o文件 
$(OBJS) : libadd.cc
	$(CC) $(CFLAGS) -I ./ -fpic -c $< -o $@

# 3. 生成动态库文件
$(LIB) : $(OBJS)
	rm -f $@
	g++ $(OBJS) -shared -o $@ 
	rm -f $(OBJS)


# 4. 删除中间过程生成的文件 
clean:
	rm -f $(OBJS) $(TARGET) $(LIB)

# 5.安装文件
install:
	mkdir -p /usr/lib/add
	cp $(LIB) /usr/lib/add
	cp $(HEADER) /usr/include