TARGET = Network
SRC = Network.cpp 


CC := g++
LIBS := -lpthread 
LIBS_DIR = 
LDFLAGS :=  -Wall -g 
INCLUDE_DIRS = -I. -I/opt/codac/include/ -I../include 
LINK = 

TARGET_DIR = ./../target/

all: Network

$(TARGET): $(SRC) 
	$(CC) $(LIBS_DIR) $(LIBS) $(LDFLAGS) $(INCLUDE_DIRS) $(LINK) -o  $(TARGET_DIR)/$(TARGET)  $(SRC) 

clean: 
	rm $(TARGET)