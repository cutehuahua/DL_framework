base_matrix.o : base_matrix.cpp
	g++ base_matrix.cpp -c -std=c++11
matrix.o : matrix.cpp
	g++ matrix.cpp -c -std=c++11
node.o : node.cpp
	g++ node.cpp -c -std=c++11
nn.o : nn.cpp
	g++ nn.cpp -c -std=c++11
loss.o : loss.cpp
	g++ loss.cpp -c -std=c++11
dataloader.o : dataloader.cpp
	g++ dataloader.cpp -c -std=c++11 
test : test.cpp matrix.o node.o base_matrix.o nn.o loss.o dataloader.o
	g++ -o test test.cpp matrix.o node.o base_matrix.o nn.o loss.o dataloader.o -std=c++11 -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs


