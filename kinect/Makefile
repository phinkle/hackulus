CC=g++
CFLAGS=-c -O3
OPENCV=`pkg-config --cflags --libs opencv`

all: register register-tests

register: main.o icp.o
	$(CC) $(OPENCV) main.o icp.o -o register

register-tests: main-tests.o icp.o
	$(CC) $(OPENCV) main-tests.o icp.o -o register-tests

main.o: main.cpp 
	$(CC) $(CFLAGS) $(OPENCV) main.cpp

main-tests.o: main-tests.cpp
	$(CC) $(CFLAGS) $(OPENCV) main-tests.cpp

icp.o: icp.cpp
	$(CC) $(CFLAGS) $(OPENCV) icp.cpp

clean:
	rm -rf *.o register register-test
