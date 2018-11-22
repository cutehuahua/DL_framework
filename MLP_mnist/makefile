all : mnist
core.o : core.cpp
	g++ core.cpp -c -std=c++11
mnist.o : mnist.cpp
	g++ mnist.cpp -c -std=c++11
dataloader.o : dataloader.cpp
	g++ dataloader.cpp -c -std=c++11 -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs
mnist : core.o mnist.o dataloader.o
	g++ -o mnist dataloader.o mnist.o core.o -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs