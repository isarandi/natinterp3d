CC = gcc
CFLAGS = -Wall -O3

PROG = natural
OBJS = natural.o delaunay.o utils.o kdtree.o
SRCS = natural.c natural.h delaunay.c delaunay.h utils.c utils.h kdtree.c kdtree.h
invent: $(OBJS)
	$(CC) $(CFLAGS) -lm -fopenmp -lpthread -o  $(PROG) $(OBJS) $(LDLIBS)


.c.o:
	$(CC) $(CFLAGS) -c $*.c

clean:
	rm -f $(OBJS) $(PROG)

depend:	
	makedepend -- $(CFLAGS) -- $(SRCS)

