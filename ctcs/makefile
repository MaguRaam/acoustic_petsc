include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

run: main
	mpirun -n 8 ./main

all: main

main : main.o
	-${CLINKER} -o main main.o ${PETSC_LIB}
	${RM} main.o

distclean:
	rm -r plot/* main