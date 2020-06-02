CC = gcc
CFLAGS = -ansi -Wall

network: network.o misc_functions.o
	$(CC) $(CFLAGS) -o network network.o misc_functions.o -lm

network.o: network.c
	$(CC) $(CFLAGS) -c network.c

misc_functions.o: misc_functions.c
	$(CC) $(CFLAGS) -c misc_functions.c

all: network

clean:
	rm network network.o misc_functions.o
