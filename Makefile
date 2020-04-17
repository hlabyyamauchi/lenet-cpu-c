all: cnnfunc.c lenet.c
	gcc -lm -o lenet lenet.c cnnfunc.c
clean:
	rm lenet
