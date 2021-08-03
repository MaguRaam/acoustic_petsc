include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


TARGET = wave

all: $(TARGET)

$(TARGET) : $(TARGET).o
	-${CLINKER} -o $(TARGET) $(TARGET).o  ${PETSC_LIB}
	$(RM) $(TARGET).o

distclean :
	$(RM) $(TARGET) *.vts