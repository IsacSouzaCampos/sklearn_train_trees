all: program

program: main.o
	g++ -g3 main.o -o program -fopenmp

main.o: main.cpp
	g++ -g3 -c main.cpp -fopenmp

clean:
	rm *.o
