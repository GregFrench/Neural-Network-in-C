CC = gcc
CFLAGS = -ansi -Wall

main: main.o misc_functions.o mnist_loader.o network.o
	$(CC) $(CFLAGS) -o main main.o misc_functions.o mnist_loader.o network.o -lm

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

misc_functions.o: misc_functions.c
	$(CC) $(CFLAGS) -c misc_functions.c

mnist_loader.o: mnist_loader.c
	$(CC) $(CFLAGS) -c mnist_loader.c

network.o: network.c
	$(CC) $(CFLAGS) -c network.c

all: main

clean:
	rm main main.o misc_functions.o mnist_loader.o network.o
