## CS231n
### This project is done based on Anaconda3(python3.6) and Win10 instead of Linux
Therefore, there are some things should be noticed  
  
First, the dataset should be manually down loaded from  https://www.cs.toronto.edu/~kriz/cifar.html and added to the /cs231n/datasets fold

Second, in assignment2(fast_layer), you might encounter some problem with cython in Windows(Unable to find vcvarsall.bat), which can be solved following the instructions in https://link.zhihu.com/?target=https%3A//blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/%23comments. Then you can run Anaconda prompt and complie the cython file.
