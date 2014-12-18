#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <cstdlib>

__constant__ unsigned int d_lookup[256];

int get_one(unsigned int value);

struct MyBitMap {
	unsigned int *bits;
	int x,y;
	unsigned long long int size;

	MyBitMap(int row, int col) {
		int intCols = (col+31)/32;
		size = (unsigned long long int)row*(unsigned long long int)intCols;
		//printf("Need size: %llu, row: %d, cols: %d \n", size, row, intCols);
		bits = new unsigned int[size];
		x = row;
		y = intCols;
		for(int i=0; i<row*intCols; i++)
			bits[i] = 0;
	}
	~MyBitMap() {
		delete [] bits;
	}
	int getSize(){
		return x*y;
	}
	unsigned int *getPointer() {
		return bits;
	}
	int getRow() {
		return x;
	}
	int getCol() {
		return y;
	}
	void setRow(int row1, unsigned int *second, int row2) {
		for(int i=0; i<y; i++) {
			bits[row1*y+i] = second[row2*y+i];
		}
	}
	void resize(int row, int col) {
		delete [] bits;
		int intCols = (col+31)/32;
		size = (unsigned long long int)row*(unsigned long long int)intCols;
		//printf("Need size: %llu \n", size);
		bits = new unsigned int[size];
		x = row;
		y = intCols;
		for(int i=0; i<row*intCols; i++)
			bits[i] = 0;
	}
	unsigned int getInt(int row, int colInt) {
		return bits[row*y+colInt];
	}
	void setInt(int row, int colInt, unsigned int value) {
		bits[row*y+colInt] = value;
	}
	void setBit(int row, int col) {
		int i = row*y+col/32;
		unsigned int flag = 1;
		flag = flag<<(31-col%32);
		bits[i] = bits[i] | flag;
	}
	void clearBit(int row, int col) {
		int i = row*y+col/32;	
		unsigned int flag = 1;
		flag = flag<<(31-col%32);
		if((bits[i]&flag) != 0)
			bits[i] = bits[i] - flag;
	}
	unsigned int getBit(int row, int col) {
		int i = row*y+col/32;
		unsigned int flag = 1;
		flag = flag<<(31-col%32);
		if((flag&bits[i]) == 0)
			return 0;
		else
			return 1;	
	}
	void print(int row) {
		for(int i=0; i<y; i++)
			std::cout<<bits[row*y+i]<<" ";
	}
};

__global__ void count_ones(unsigned int *d_itemBitmap, unsigned int *d_bitmap, int numItem, int numTxn, int support)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	for (int i=idx; i<numItem; i += blockDim.x*gridDim.x) {
		int count = 0;
		int colInt = (numTxn+31)/32;
		for(int j=0; j<colInt; ++j){
			unsigned int temp = d_bitmap[i*colInt+j];
			unsigned int one = 255;
			one = one&temp;
			temp=temp>>8;
			unsigned int two = 255;
			two = two&temp;
			temp=temp>>8;
			unsigned int three = 255;
			three = three&temp;
			unsigned int four = temp>>8; 
			count += d_lookup[one]+d_lookup[two]+d_lookup[three]+d_lookup[four];
		}
		if(count >= support){
			int itemMapCol = (numItem+1+32)/32;	
			int index = itemMapCol*i+itemMapCol-1;
			unsigned int flag = 1;
			flag = flag<<(31-numItem%32);
			d_itemBitmap[index] = d_itemBitmap[index] | flag;
		}
	}
}

__global__ void testSupport(unsigned int *pairs, unsigned int *d_parent_transactions, unsigned int *d_child_transactions, unsigned int *d_child_items, int numItem, int support, int numTxn, int numChild)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	for (int i=idx; i<numChild; i += blockDim.x*gridDim.x) {
		int count = 0;
		int colTxn = (numTxn+31)/32;
		int colItem = (numItem+32)/32;
		for(int j=0; j<colTxn; ++j) {
			int a = pairs[2*i];
			int b = pairs[2*i+1];
			unsigned int temp = d_parent_transactions[a*colTxn+j] & d_parent_transactions[b*colTxn+j];
			d_child_transactions[i*colTxn+j]=temp;
			
			unsigned int one = 255;
			one = one&temp;
			temp=temp>>8;
			unsigned int two = 255;
			two = two&temp;
			temp=temp>>8;
			unsigned int three = 255;
			three = three&temp;
			unsigned int four = temp>>8; 
			count += d_lookup[one]+d_lookup[two]+d_lookup[three]+d_lookup[four];
		}
		if(count >= support) {
			int indexHere = colItem*(i+1)-1; 			
			unsigned int flag=1;
			flag = flag<<(31-numItem%32);
			d_child_items[indexHere] = d_child_items[indexHere] | flag;
		}
	}
}

__global__ void generateNext(unsigned int *pairs, unsigned int *d_parent, unsigned int *d_child, int itemSize, int itemNum, int size, int rowsItem)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	for (int i=idx; i<size; i += blockDim.x*gridDim.x) {
		int a=0;
		int b;
		int newI = i+1;
		int temp = rowsItem-1;
		while(newI>temp) {
			a++;
			newI -= temp;
			temp--;
		}
		b=a+newI;
		int colInt = (itemNum+32)/32;
		int equal = itemSize-2;
		for(int p=0; p<colInt; p++) {
			unsigned int aParent = d_parent[a*colInt+p];
			unsigned int bParent = d_parent[b*colInt+p];
			//printf("a: %d, b: %d, avalue: %u, bvalue: %u, p: %d, equal: %d\n",a,b, aParent, bParent, p, equal);
			unsigned int flag = 1;
			flag = flag<<31;
			int satisfy=1;
			for(int q=0; q<32; q++) {
				if(equal==0) {
					satisfy = 2;
					break;
				}
				if((aParent&flag) != (bParent&flag)){
					satisfy = 0;
					break;
				}
				else {
					if((aParent&flag)!=0)
						--equal;
				}
				flag = flag>>1;
			}
			if(satisfy==2) {
				for(int m=0; m<colInt; m++){
					unsigned int aNewParent = d_parent[a*colInt+m];
					unsigned int bNewParent = d_parent[b*colInt+m];
					d_child[i*colInt+m] = aNewParent | bNewParent;
				}
				int indexHere = (i+1)*colInt-1;
				unsigned int flag=1;
				flag = flag<<(31-itemNum%32);	
				d_child[indexHere] = d_child[indexHere] | flag;	
				pairs[i*2] = a;
				pairs[i*2+1] = b;
				//printf("satisfied a: %d, b: %d , d_childlast: %u \n",a, b, d_child[indexHere]);
				break;
			}
			if(satisfy==0){
				int indexHere = (i+1)*colInt-1;
				d_child[indexHere] = 0;	
				break;
			}
		}
	}
}

int main(int argc, char *argv[])
{
	std::ifstream input_file(argv[1]);
	int numBlock = atoi(argv[2]);
	int numThreads = atoi(argv[3]);
	float support_ratio=0.01;
	int tnx, numItem;
	input_file>>tnx>>numItem;
	float totalTime = 0;
	MyBitMap bitmap(numItem, tnx);
	int support = tnx*support_ratio;
	std::string tempLine;
	std::getline(input_file, tempLine);
	for(int i=0; i<tnx; i++) {
		std::string oneline;
		std::getline(input_file, oneline);
		std::istringstream items(oneline);
		int item;
		while(items>>item){	
			if (item<=numItem && item >0)
				bitmap.setBit(item-1, i);	
		}
		items.clear();
	}
	MyBitMap itemBitmap(numItem, numItem+1);
	for(int i=0; i<numItem; i++) {
		itemBitmap.setBit(i, i);
	}
		
	int lookup[256];
	for(unsigned int i=0; i<256; i++) {
		lookup[i]=get_one(i);
	}
	cudaMemcpyToSymbol(d_lookup, lookup, sizeof(int)*256);
	unsigned int *d_bitmap, *d_itemBitmap;
	cudaMalloc(&d_bitmap, bitmap.getSize()*sizeof(unsigned int));
	cudaMalloc(&d_itemBitmap, itemBitmap.getSize()*sizeof(unsigned int));

	cudaMemcpy(d_bitmap, bitmap.getPointer(), bitmap.getSize()*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_itemBitmap, itemBitmap.getPointer(), itemBitmap.getSize()*sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	count_ones<<<numBlock, numThreads>>>(d_itemBitmap, d_bitmap, numItem, tnx, support);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	totalTime+=milliseconds;
	std::cout<<"Init time: "<<milliseconds<<"--------------------------"<<std::endl;

	cudaMemcpy(bitmap.getPointer(),d_bitmap, bitmap.getSize()*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(itemBitmap.getPointer(), d_itemBitmap, itemBitmap.getSize()*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_bitmap);
	cudaFree(d_itemBitmap);
	int oldCount = numItem;
	int newCount = 0;	
	for(int i=0; i<numItem; i++) {
		if(itemBitmap.getBit(i, numItem) == 1)
			newCount++;
	}
	
	int tnxCol = (tnx+31)/32;
	int itemCol = (numItem+32)/32;
	int itemSize = 1;
	while(newCount > 1) {
		std::cout<<std::endl<<"new itemSize: "<<itemSize<<"  newCount: "<<newCount<<std::endl<<std::endl;
		itemSize++;
		MyBitMap newBitmap(newCount, tnx);
		MyBitMap newItemmap(newCount, numItem+1);
		int j=0;
		for(int i=0; i<oldCount; i++) {
			if(itemBitmap.getBit(i, numItem) == 1) {
				newBitmap.setRow(j, bitmap.getPointer(), i);	
				newItemmap.setRow(j, itemBitmap.getPointer(), i);
				newItemmap.clearBit(j, numItem);
				j++;
			}
		}
		int possibleNextChild = (newCount)*(newCount-1)/2;	
		unsigned int *d_pairs, *d_parent, *d_child;
		cudaMalloc(&d_pairs, 2*possibleNextChild*sizeof(unsigned int));		
		cudaMalloc(&d_parent, newCount*sizeof(unsigned int)*itemCol);
		cudaMalloc(&d_child, possibleNextChild*itemCol*sizeof(unsigned int));
		printf("Device Variable alloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
		cudaMemcpy(d_parent, newItemmap.getPointer(), newItemmap.getSize()*sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaEventRecord(start);
		generateNext<<<numBlock, numThreads>>> (d_pairs, d_parent, d_child, itemSize, numItem, possibleNextChild, newCount);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		totalTime+=milliseconds;
		std::cout<<"generate time: "<<milliseconds<<"--------------------------"<<std::endl;
		unsigned int *pairs = new unsigned int[2*possibleNextChild];
		MyBitMap child(possibleNextChild, numItem+1);
		cudaError_t error1 = cudaMemcpy(pairs, d_pairs, 2*possibleNextChild*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaError_t error2 = cudaMemcpy(child.getPointer(), d_child, itemCol*possibleNextChild*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//printf("Error1: %s\n", cudaGetErrorString(error1));
		//printf("Error2: %s\n", cudaGetErrorString(error2));
		printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));
		cudaFree(d_child);
		cudaFree(d_pairs);
		cudaFree(d_parent);
		int usefulChild=0;
		for(int m=0; m<possibleNextChild; m++) {
			if(child.getBit(m,numItem) == 1)
				usefulChild++;
		}
		unsigned int *pairsGen = new unsigned int[2*usefulChild];
		std::cout<<std::endl<<"usefulChild:"<<usefulChild<<std::endl<<std::endl;
		itemBitmap.resize(usefulChild, numItem+1);
		j=0;
		for(int m=0; m<possibleNextChild; m++) {
			if(child.getBit(m, numItem) == 1) {
				itemBitmap.setRow(j, child.getPointer(), m);
				itemBitmap.clearBit(j, numItem);
				pairsGen[j*2]=pairs[2*m];
				pairsGen[j*2+1]=pairs[2*m+1];
				++j;
			}
		}
		delete []pairs;
		unsigned int *d_parent_tnx, *d_child_tnx, *d_child_item;
		cudaMalloc(&d_pairs, 2*usefulChild*sizeof(unsigned int));		
		cudaMalloc(&d_parent_tnx, newCount*sizeof(unsigned int)*tnxCol);		
		cudaMalloc(&d_child_tnx, usefulChild*sizeof(unsigned int)*tnxCol);		
		cudaMalloc(&d_child_item, usefulChild*sizeof(unsigned int)*itemCol);
		cudaMemcpy(d_pairs, pairsGen, 2*usefulChild*sizeof(unsigned int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_parent_tnx,newBitmap.getPointer() , newCount*sizeof(unsigned int)*tnxCol,cudaMemcpyHostToDevice);
		cudaMemcpy(d_child_item,itemBitmap.getPointer() , usefulChild*sizeof(unsigned int)*itemCol,cudaMemcpyHostToDevice);
		cudaEventRecord(start);
		testSupport<<<numBlock, numThreads>>> (d_pairs, d_parent_tnx, d_child_tnx, d_child_item, numItem, support, tnx, usefulChild);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		totalTime+=milliseconds;
		std::cout<<"test time: "<<milliseconds<<"--------------------------"<<std::endl;
		bitmap.resize(usefulChild, tnx);
		cudaMemcpy(itemBitmap.getPointer(), d_child_item, usefulChild*sizeof(unsigned int)*itemCol, cudaMemcpyDeviceToHost);
		cudaMemcpy(bitmap.getPointer(), d_child_tnx, usefulChild*sizeof(unsigned int)*tnxCol, cudaMemcpyDeviceToHost);
		newCount = 0;
		for(int m=0; m<usefulChild; m++) {
			if(itemBitmap.getBit(m, numItem) == 1)
				newCount++;
		}
		oldCount = usefulChild;
		cudaFree(d_pairs);
		cudaFree(d_parent_tnx);
		cudaFree(d_child_tnx);
		cudaFree(d_child_item);
		delete[] pairsGen;
	}	
	std::cout<<"total time: "<<totalTime<<" milliseconds--------------------------"<<std::endl;
	return 0;
}
int get_one(unsigned int value){
	int count = 0;
	unsigned int flag = 1;
	for(int i=0; i<8; i++) {
		if((value&flag) == flag)
			++count;
		value = value>>1;
	}
	return count;
}
