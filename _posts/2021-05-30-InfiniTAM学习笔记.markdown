idx: 1

ITMLib->Engines->Swapping->Shared->ITMSwappingEngine_Shared.h

combineVoxelDepthInformation()

用途：更新TSDF的每个体素的权重和TSDF值。newW、newF就是权重和TSDF值的意思。

```c
template<class TVoxel>
_CPU_AND_GPU_CODE_ inline void combineVoxelDepthInformation(const CONSTPTR(TVoxel) & src, DEVICEPTR(TVoxel) & dst, int maxW)
{
	int newW = dst.w_depth;
	int oldW = src.w_depth;
	float newF = TVoxel::valueToFloat(dst.sdf);
	float oldF = TVoxel::valueToFloat(src.sdf);

	if (oldW == 0) return;

	newF = oldW * oldF + newW * newF;
	newW = oldW + newW;
	newF /= newW;
	newW = MIN(newW, maxW);

	dst.w_depth = newW;
	dst.sdf = TVoxel::floatToValue(newF);
}
```



idx: 2

ITMLib->Engines->Swapping->Shared->ITMSwappingEngine_CPU.tpp

IntegrateGlobalIntoLocal()

用途：因为InfiniTAM采用了将体素块分批处理的方式，全局数组存放在更大的存储空间中（目前还没摸清是什么, 就先理解为host中存储），device中存储的是局部的体素块，局部的体素块处理完以后，需要和全局的对应的体素块进行融合。

```c
template<class TVoxel>
void ITMSwappingEngine_CPU<TVoxel, ITMVoxelBlockHash>::IntegrateGlobalIntoLocal(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState *renderState)
```

设置指针指向整个程序中最重要的数据结构：scene的globalCache,全局的体素块就存放在这里

```
ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;
```

获取同样重要的数据结构：用于全局索引的哈希表

```
ITMHashEntry *hashTable = scene->index.GetEntries();
```

获取每个体素块（体素块是一个8x8x8的体素组成的大块）目前的内存状态,0表示在host，1表示在host和device，2表示在device，需要保存。参数false表示不使用GPU

```
ITMHashSwapState *swapStates = globalCache->GetSwapStates(false);
```

syncedVoxelBlocks是一个通过遍历全局体素数组后将，得到的同步好的体素连续拼接成的新数组

```
TVoxel *syncedVoxelBlocks_local = globalCache->GetSyncedVoxelBlocks(false);
```

hasSyncedData_local记录是否已经同步过，neededEntryIDs_local存放syncedVoxelBlocks中的体素的全局索引

```
bool *hasSyncedData_local = globalCache->GetHasSyncedData(false);
int *neededEntryIDs_local = globalCache->GetNeededEntryIDs(false);
```

localVBA表示local voxel block address，应该是device维护的局部体素数组

```
TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
```

接下来的更新过程是：

srcVB指向存放全局数组体素信息的数组，因为syncedVoxelBlocks_local存放的是连续查找全局数组后将同步过的体素块按顺序放入srcVB的，因此可以直接通过递增i线性查找。

dstVB指向存放局部体素信息的数组，因此针对于srcVB指向的体素块，需要进行哈希表的映射进行定位, 具体方式为从依次存放全局索引的neededEntryIDs_local数组中取出索引，然后使用哈希映射。哈希映射的方向是从全局索引映射到局部索引。

*不过这里我有一点搞不明白——如何保证扫描到的全局索引一定可以找到局部索引呢？（已解决，idx3）*

```
int entryDestId = neededEntryIDs_local[i];
hashTable[entryDestId].ptr;
```

接下来就是通过访问一个大块中的512个体素块，更新每一个块的信息。

```
for (int vIdx = 0; vIdx < SDF_BLOCK_SIZE3; vIdx++)
{
	CombineVoxelInformation<TVoxel::hasColorInformation, TVoxel>::compute(srcVB[vIdx], dstVB[vIdx], maxW);
}
```

```
for (int i = 0; i < noNeededEntries; i++)
	{
		int entryDestId = neededEntryIDs_local[i];

		if (hasSyncedData_local[i])
		{
			TVoxel *srcVB = syncedVoxelBlocks_local + i * SDF_BLOCK_SIZE3;
			TVoxel *dstVB = localVBA + hashTable[entryDestId].ptr * SDF_BLOCK_SIZE3;

			for (int vIdx = 0; vIdx < SDF_BLOCK_SIZE3; vIdx++)
			{
				CombineVoxelInformation<TVoxel::hasColorInformation, TVoxel>::compute(srcVB[vIdx], dstVB[vIdx], maxW);
			}
		}

		swapStates[entryDestId].state = 2;
	}
```



idx: 3

ITMLib->Engines->Swapping->Shared->ITMSwappingEngine_CPU.tpp

LoadFromGlobalMemory()

用途：检查全局数组中同步过的块，进行再次同步。这就是论文中的swapin的过程

```
template<class TVoxel>
int ITMSwappingEngine_CPU<TVoxel, ITMVoxelBlockHash>::LoadFromGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
```

访问所有的块，如果swapState是1，也即既在device也在host，就说明正在更新这个块（目前device中正在处理这个块）这也解释了idx2中的疑惑（保证了全局数组中找到的块device中一定存在），然后将这个块添加进neededEntryIDs_local中，如果超过TSDF能够处理的块的数量就结束。

````
	int noNeededEntries = 0;
	for (int entryId = 0; entryId < noTotalEntries; entryId++)
	{
		if (noNeededEntries >= SDF_TRANSFER_BLOCK_NUM) break;
		if (swapStates[entryId].state == 1)
		{
			neededEntryIDs_local[noNeededEntries] = entryId;
			noNeededEntries++;
		}
	}
````

上一步是将所有的索引存进去了，这一步将体素也拷贝进syncedVoxelBlocks_global中，然后idx2中会根据这个数组获得需要更新local数组的信息。

````
if (noNeededEntries > 0)
	{
		memset(syncedVoxelBlocks_global, 0, noNeededEntries * SDF_BLOCK_SIZE3 * sizeof(TVoxel));
		memset(hasSyncedData_global, 0, noNeededEntries * sizeof(bool));
		for (int i = 0; i < noNeededEntries; i++)
		{
			int entryId = neededEntryIDs_global[i];

			if (globalCache->HasStoredData(entryId))
			{
				hasSyncedData_global[i] = true;
				memcpy(syncedVoxelBlocks_global + i * SDF_BLOCK_SIZE3, globalCache->GetStoredVoxelBlock(entryId), SDF_BLOCK_SIZE3 * sizeof(TVoxel));
			}
		}
	}

````



idx：4



```
template<class TVoxel>
void ITMSwappingEngine_CPU<TVoxel, ITMVoxelBlockHash>::SaveToGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState *renderState)
```

