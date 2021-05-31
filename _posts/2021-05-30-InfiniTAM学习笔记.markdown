## swap

### idx: 1

#### ITMLib->Engines->Swapping->Shared->ITMSwappingEngine_Shared.h

#### combineVoxelDepthInformation()

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



### idx: 2

#### ITMLib->Engines->Swapping->Shared->ITMSwappingEngine_CPU.tpp

#### IntegrateGlobalIntoLocal()

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



### idx: 3

#### ITMLib->Engines->Swapping->Shared->ITMSwappingEngine_CPU.tpp

#### LoadFromGlobalMemory()

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



### idx：4

#### ITMLib->Engines->Swapping->Shared->ITMSwappingEngine_CPU.tpp

#### SaveToGlobalMemory（）

功能:实现了swapout的功能

```
template<class TVoxel>
void ITMSwappingEngine_CPU<TVoxel, ITMVoxelBlockHash>::SaveToGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState *renderState)
```

首先同样遍历每个全局数组中的块，看看是否满足swapState为2，即仅在device中需要保存，以及localPtr>0，即保证能够通过哈希表找到局部数组的块，以及entriesVisibletype为0*（这个我猜测是判断如果一个体素块被表面包裹的话，就没有被更新，没必要存回host，待解决）*

```
for (int entryDestId = 0; entryDestId < noTotalEntries; entryDestId++)
	{
		if (noNeededEntries >= SDF_TRANSFER_BLOCK_NUM) break;

		int localPtr = hashTable[entryDestId].ptr;
		ITMHashSwapState &swapState = swapStates[entryDestId];

		if (swapState.state == 2 && localPtr >= 0 && entriesVisibleType[entryDestId] == 0)
```

然后将满足条件的块存回host，值得注意的一点是这一步先存回缓冲区syncedVoxelBlock（不直接存host）

```
			TVoxel *localVBALocation = localVBA + localPtr * SDF_BLOCK_SIZE3;

			neededEntryIDs_local[noNeededEntries] = entryDestId;

			hasSyncedData_local[noNeededEntries] = true;
			memcpy(syncedVoxelBlocks_local + noNeededEntries * SDF_BLOCK_SIZE3, localVBALocation, SDF_BLOCK_SIZE3 * sizeof(TVoxel));

			swapStates[entryDestId].state = 0;
```

然后将被存回的块的每一个体素替换为新的未初始化的体素，每一个块存回host（暂时是缓冲区)以后，这个local的块就变得空闲，然后将noAllocatedVoxelEntries加1，表示空闲的块的数量，voxelAllocationList[vbaIdx + 1] = localPtr则记录空闲的块对应local memory的位置，以便随后分配新的块。

```
int vbaIdx = noAllocatedVoxelEntries;
if (vbaIdx < SDF_BUCKET_NUM - 1)
{
    noAllocatedVoxelEntries++;
    voxelAllocationList[vbaIdx + 1] = localPtr;
    hashTable[entryDestId].ptr = -1;

    for (int i = 0; i < SDF_BLOCK_SIZE3; i++) localVBALocation[i] = TVoxel();
}
```



### idx：5

#### ITMLib->Engines->Swapping->Shared->ITMSwappingEngine_CPU.tpp

#### CleanLocalMemory（）

用途：和idx4一样，但是这个是将所有的local都清空了。

```
template<class TVoxel>
void ITMSwappingEngine_CPU<TVoxel, ITMVoxelBlockHash>::CleanLocalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState *renderState)
```



## Tracker（Depth Tracker）

Depth Tracker用来通过深度图信息判断位姿，通过使用列文伯格马夸尔特方法进行非线性优化。

具体的思路是：



![image-20210530192330785](https://raw.githubusercontent.com/zjnyly/blog/main/img/20210530193209.png)

这里使用了ICP方法（ICP还没仔细看）大概通俗的解释是：

由于使用TSDF方式进行三维建模，TSDF的思路是预想一个物体的形状，然后不断地将其修正为真实的形状，因此需要有一个观念就是我们使用TSDF维护的三维模型就是最真实的，无论是否真的真实，也将其当作真实值作为参照。

在InfiniTAM中，Depth Tracker需要以下几种信息

1. TSDF模型
2. 上一次迭代生成的深度图（认为是真实的）*（具体是怎么生成的还需要解决）*
3. 预估的相机位姿

然后我们通过建立一个最优化问题，优化的是相机位姿，方式为：

首先通过将深度图上的2D点

![image-20210530193201886](https://raw.githubusercontent.com/zjnyly/blog/main/img/20210530193204.png)

通过相机内参配合深度信息投影为相机坐标系下的3D点P(x),然后使用估计的R和t将其转为世界坐标系下的3D点，

接着通过Raycasting*（这个需要进一步研究）*将相片上的点投影到TSDF模型上，如果射线和TSDF模型的表面相交（找到zero-crossing），就找到了点P^{-},这个过程应该是N（）映射在起作用，而P^{-}实际上应该是P重新反投影到相机平面，通过Raycasting形成的（这里我猜是为了强调P这个3D点才有了P^{-}和P，直接用2D的点x也可以直接表示）

![image-20210530192315731](https://raw.githubusercontent.com/zjnyly/blog/main/img/20210530192317.png)

最后就是求解出最好的R和t，使得这个能量函数能化为最小，方式为

raycasting得到的点和使用R和t估计的点的距离点乘TSDF上该点的法线，乘以法线应该是为了保证数据的归一化。

### idx： 6

#### ITMLib->Trackers->Interface->ITMDepthTracker.cpp

#### TrackCamera（）

用途：用于更新R和t

```
void ITMDepthTracker::TrackCamera(ITMTrackingState *trackingState, const ITMView *view)
```

返回值是可以用来判断的点的数目，函数的功能是计算根据当前估计的新的位姿估计误差，计算nabla算子，hessian矩阵（idx7)

```
noValidPoints_new = this->ComputeGandH(f_new, nabla_new, hessian_new, approxInvPose);
```

从idx6返回，如果结果变差，就回滚，否则更新，更新时hessian_new[i] / noValidPoints_new是取平均（idx6中把所有点的hessian矩阵加一块了）

```
if ((noValidPoints_new <= 0) || (f_new > f_old)) {
				trackingState->pose_d->SetFrom(&lastKnownGoodPose);
				approxInvPose = trackingState->pose_d->GetInvM();
				lambda *= 10.0f;
			}
			else {
				lastKnownGoodPose.SetFrom(trackingState->pose_d);
				f_old = f_new;
				noValidPoints_old = noValidPoints_new;

				for (int i = 0; i < 6 * 6; ++i) hessian_good[i] = hessian_new[i] / noValidPoints_new;
				for (int i = 0; i < 6; ++i) nabla_good[i] = nabla_new[i] / noValidPoints_new;
				lambda /= 10.0f;
			}
```

这一步使用LM优化，计算优化公式(H+\lambda I)\delta x_{k} = g中的(H+\lambda I)

```
for (int i = 0; i < 6 * 6; ++i) A[i] = hessian_good[i];
			for (int i = 0; i < 6; ++i) A[i + i * 6] *= 1.0f + lambda;
```

接下来就是线性优化的部分

```
ComputeDelta(step, nabla_good, A, iterationType != TRACKER_ITERATION_BOTH);
ApplyDelta(approxInvPose, step, approxInvPose);
trackingState->pose_d->SetInvM(approxInvPose);
trackingState->pose_d->Coerce();
approxInvPose = trackingState->pose_d->GetInvM();

// if step is small, assume it's going to decrease the error and finish
if (HasConverged(step)) break;
```

如果收敛就结束，其中ComputeDelta（idx11)

```
ComputeDelta(step, nabla_good, A, iterationType != TRACKER_ITERATION_BOTH);
```

以及ApplyDelta（idx12）

```
ApplyDelta(approxInvPose, step, approxInvPose);
```

最后如果收敛，本次位姿判断就结束了。



### idx: 7

#### ITMLib->Trackers->CPU->ITMDepthTracker_CPU.cpp

#### ComputeGandH()

功能：根据当前估计的新的位姿估计误差，计算nabla算子，hessian矩阵

```
int ITMDepthTracker_CPU::ComputeGandH(float &f, float *nabla, float *hessian, Matrix4f approxInvPose)
```

这里枚举图片中的每一个点，按照不同的计算模式（只考虑旋转、只考虑平移，二者都考虑）选择不同的实例，计算每一个点的localHessian和localNabla（idx8）

```
for (int y = 0; y < viewImageSize.y; y++) for (int x = 0; x < viewImageSize.x; x++)
	{
		float localHessian[6 + 5 + 4 + 3 + 2 + 1], localNabla[6], localF = 0;

		for (int i = 0; i < noPara; i++) localNabla[i] = 0.0f;
		for (int i = 0; i < noParaSQ; i++) localHessian[i] = 0.0f;

		bool isValidPoint;
        
		switch (iterationType)
		{
		case TRACKER_ITERATION_ROTATION:
			isValidPoint = computePerPointGH_Depth<true, true>(localNabla, localHessian, localF, x, y, depth[x + y * viewImageSize.x], viewImageSize,
				viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh[levelId]);
			break;
		case TRACKER_ITERATION_TRANSLATION:
			isValidPoint = computePerPointGH_Depth<true, false>(localNabla, localHessian, localF, x, y, depth[x + y * viewImageSize.x], viewImageSize,
				viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh[levelId]);
			break;
		case TRACKER_ITERATION_BOTH:
			isValidPoint = computePerPointGH_Depth<false, false>(localNabla, localHessian, localF, x, y, depth[x + y * viewImageSize.x], viewImageSize,
				viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh[levelId]);
			break;
		default:
			isValidPoint = false;
			break;
		}
```

从idx7返回后，判断这个点是否是有效点，如果是，就将该点的hessian加到该帧的sumhessian矩阵中

```
if (isValidPoint)		{			noValidPoints++; sumF += localF;			for (int i = 0; i < noPara; i++) sumNabla[i] += localNabla[i];			for (int i = 0; i < noParaSQ; i++) sumHessian[i] += localHessian[i];		}
```

这里将idx7里的Hessian矩阵补全，并生成尚未平均的整体hessian

```
for (int r = 0, counter = 0; r < noPara; r++) for (int c = 0; c <= r; c++, counter++) hessian[r + c * 6] = sumHessian[counter];	for (int r = 0; r < noPara; ++r) for (int c = r + 1; c < noPara; c++) hessian[r + c * 6] = hessian[c + r * 6];
```





### idx：8

#### ITMLib->Trackers->Shared->ITMDepthTracker_Shared.h

#### computePerPointGH_Depth（）

功能：计算localNabla和locaHessian和localF，补全Hessian矩阵

```
template<bool shortIteration, bool rotationOnly>_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth(THREADPTR(float) *localNabla, THREADPTR(float) *localHessian, THREADPTR(float) &localF,	const THREADPTR(int) & x, const THREADPTR(int) & y,	const CONSTPTR(float) &depth, const CONSTPTR(Vector2i) & viewImageSize, const CONSTPTR(Vector4f) & viewIntrinsics, const CONSTPTR(Vector2i) & sceneImageSize,	const CONSTPTR(Vector4f) & sceneIntrinsics, const CONSTPTR(Matrix4f) & approxInvPose, const CONSTPTR(Matrix4f) & scenePose, const CONSTPTR(Vector4f) *pointsMap,	const CONSTPTR(Vector4f) *normalsMap, float distThresh)
```

这句话调用真正的计算函数(idx 9)

```
bool ret = computePerPointGH_Depth_Ab<shortIteration,rotationOnly>(A, b, x, y, depth, viewImageSize, viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh);
```

从idx9返回以后，使用计算的A矩阵（jacobian矩阵）计算nabla算子，并使用 J^{T}J拟合Hessian矩阵（这里只填充了一半）

```
for (int r = 0, counter = 0; r < noPara; r++)	{		localNabla[r] = b * A[r];		for (int c = 0; c <= r; c++, counter++) localHessian[counter] = A[r] * A[c];	}
```





### idx: 9

#### ITMLib->Trackers->Shared->ITMDepthTracker_Shared.h

#### computePerPointGH_Depth_Ab()

功能：真正计算localNabla和locaHessian和localF

```
template<bool shortIteration, bool rotationOnly>_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth_Ab(THREADPTR(float) *A, THREADPTR(float) &b,	const THREADPTR(int) & x, const THREADPTR(int) & y,	const CONSTPTR(float) &depth, const CONSTPTR(Vector2i) & viewImageSize, const CONSTPTR(Vector4f) & viewIntrinsics, const CONSTPTR(Vector2i) & sceneImageSize,	const CONSTPTR(Vector4f) & sceneIntrinsics, const CONSTPTR(Matrix4f) & approxInvPose, const CONSTPTR(Matrix4f) & scenePose, const CONSTPTR(Vector4f) *pointsMap,	const CONSTPTR(Vector4f) *normalsMap, float distThresh)
```

首先使用齐次坐标记录通过相机内参和深度信息投影而成的tmp3Dpoint（这是根据当前位姿计算出来的深度图上一个像素点的3D坐标估计值）

```
tmp3Dpoint.x = depth * ((float(x) - viewIntrinsics.z) / viewIntrinsics.x);	tmp3Dpoint.y = depth * ((float(y) - viewIntrinsics.w) / viewIntrinsics.y);	tmp3Dpoint.z = depth;	tmp3Dpoint.w = 1.0f;    	// transform to previous frame coordinates	tmp3Dpoint = approxInvPose * tmp3Dpoint;	tmp3Dpoint.w = 1.0f;
```

然后将3D点反投影回相机平面，后通过bilineaer插值找到通过raycast获得的TSDF模型中表面上的点的3D坐标

```
// project into previous rendered image	tmp3Dpoint_reproj = scenePose * tmp3Dpoint;	if (tmp3Dpoint_reproj.z <= 0.0f) return false;	tmp2Dpoint.x = sceneIntrinsics.x * tmp3Dpoint_reproj.x / tmp3Dpoint_reproj.z + sceneIntrinsics.z;	tmp2Dpoint.y = sceneIntrinsics.y * tmp3Dpoint_reproj.y / tmp3Dpoint_reproj.z + sceneIntrinsics.w;	if (!((tmp2Dpoint.x >= 0.0f) && (tmp2Dpoint.x <= sceneImageSize.x - 2) && (tmp2Dpoint.y >= 0.0f) && (tmp2Dpoint.y <= sceneImageSize.y - 2)))		return false;	curr3Dpoint = interpolateBilinear_withHoles(pointsMap, tmp2Dpoint, sceneImageSize);
```

使用双线性插值计算2D坐标下以TSDF为真实世界模型的法向量和P点坐标的真实值，

```
curr3Dpoint = interpolateBilinear_withHoles(pointsMap, tmp2Dpoint, sceneImageSize);corr3Dnormal = interpolateBilinear_withHoles(normalsMap, tmp2Dpoint, sceneImageSize);
```

计算diff

```
	ptDiff.x = curr3Dpoint.x - tmp3Dpoint.x;	ptDiff.y = curr3Dpoint.y - tmp3Dpoint.y;	ptDiff.z = curr3Dpoint.z - tmp3Dpoint.z;	float dist = ptDiff.x * ptDiff.x + ptDiff.y * ptDiff.y + ptDiff.z * ptDiff.z;
```

计算误差函数

```
b = corr3Dnormal.x * ptDiff.x + corr3Dnormal.y * ptDiff.y + corr3Dnormal.z * ptDiff.z;
```

接下来这部分则是计算Jacobian矩阵

```
if (shortIteration)	{		if (rotationOnly)		{			A[0] = +tmp3Dpoint.z * corr3Dnormal.y - tmp3Dpoint.y * corr3Dnormal.z;			A[1] = -tmp3Dpoint.z * corr3Dnormal.x + tmp3Dpoint.x * corr3Dnormal.z;			A[2] = +tmp3Dpoint.y * corr3Dnormal.x - tmp3Dpoint.x * corr3Dnormal.y;		}		else { A[0] = corr3Dnormal.x; A[1] = corr3Dnormal.y; A[2] = corr3Dnormal.z; }	}	else	{		A[0] = +tmp3Dpoint.z * corr3Dnormal.y - tmp3Dpoint.y * corr3Dnormal.z;		A[1] = -tmp3Dpoint.z * corr3Dnormal.x + tmp3Dpoint.x * corr3Dnormal.z;		A[2] = +tmp3Dpoint.y * corr3Dnormal.x - tmp3Dpoint.x * corr3Dnormal.y;		A[!shortIteration ? 3 : 0] = corr3Dnormal.x; A[!shortIteration ? 4 : 1] = corr3Dnormal.y; A[!shortIteration ? 5 : 2] = corr3Dnormal.z;	}
```

首先是判断只计算旋转还是旋转和平移都计算，我们这里将二者都考虑进去。

接着要理解一下这里的Jacobian所包含的项。

首先，Depth Tracker的优化问题是求解出R和t，使得整体的误差最小，这里的R和t就是我们需要优化的项，而P以及P_{-}是已知量，也即被优化的函数是F（R，t）的形式，需要对dF(R,t)/d(R)和dF(R,t)/dt求导，dF(R,t)/dt可以拆成dF(R,t)/dt.x, dF(R,t)/dt.y, dF(R,t)/dt.z,对应

```
A[!shortIteration ? 3 : 0] = corr3Dnormal.x; A[!shortIteration ? 4 : 1] = corr3Dnormal.y; A[!shortIteration ? 5 : 2] = corr3Dnormal.z;
```

dF(R,t)/d(R)又该如何计算？R是一个矩阵，无法拆分（大概这个意思），因此可以考虑使用李代数的形式表示，在李代数中，SO(3)给出映射R = exp(\phi^),也即使用\phi这个三维向量就可以表示一个旋转矩阵，对这个\phi求导以后就可以得到*(具体怎么求导还有待解决）*

```
A[0] = +tmp3Dpoint.z * corr3Dnormal.y - tmp3Dpoint.y * corr3Dnormal.z;A[1] = -tmp3Dpoint.z * corr3Dnormal.x + tmp3Dpoint.x * corr3Dnormal.z;A[2] = +tmp3Dpoint.y * corr3Dnormal.x - tmp3Dpoint.x * corr3Dnormal.y;
```

最后，A[0:5]全部求出，也即Jacobian矩阵求解结束。



### idx： 10

#### interpolateBilinear_withHoles()

功能：使用根据TSDF模型并使用生成的当前位置下的点云图/法线图，并使用bilinear插值，获取2D坐标上，根据已知点云图/法线图拟合出的最近似的P点坐标/法向量。*（需要进一步了解的是如何使用TSDF生成点云图/法线图（具体是在哪一个engine生成的））*

```
template<typename T> _CPU_AND_GPU_CODE_ inline Vector4f interpolateBilinear_withHoles(const CONSTPTR(ORUtils::Vector4<T>) *source,	const THREADPTR(Vector2f) & position, const CONSTPTR(Vector2i) & imgSize)
```



### idx: 11

#### ComputeDelta()

功能：使用Cholesky方法分解Hessian矩阵，并结合nabla计算下一步的步长

```
void ITMDepthTracker::ComputeDelta(float *step, float *nabla, float *hessian, bool shortIteration) const{	for (int i = 0; i < 6; i++) step[i] = 0;	if (shortIteration)	{		float smallHessian[3 * 3];		for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) smallHessian[r + c * 3] = hessian[r + c * 6];		ORUtils::Cholesky cholA(smallHessian, 3);		cholA.Backsub(step, nabla);	}	else	{		ORUtils::Cholesky cholA(hessian, 6);		cholA.Backsub(step, nabla);	}}
```

step是一个1x6的向量，包含旋转矩阵的李代数形式和位移向量（6个自由度）

```
void Backsub(F *result, const F *v) const		{			std::vector<F> y(size);			for (int i = 0; i < size; i++)			{				F val = v[i];				for (int j = 0; j < i; j++) val -= cholesky[j + i * size] * y[j];				y[i] = val;			}			for (int i = 0; i < size; i++) y[i] /= cholesky[i + i * size];			for (int i = size - 1; i >= 0; i--)			{				F val = y[i];				for (int j = i + 1; j < size; j++) val -= cholesky[i + j * size] * result[j];				result[i] = val;			}		}
```

BackSub起到的作用应该就是将Hessian矩阵分解*（还有待考证）*



### idx: 12

#### ApplyDelta（）

作用：制作矩阵T，用于变换位姿

```
void ITMDepthTracker::ApplyDelta(const Matrix4f & para_old, const float *delta, Matrix4f & para_new) const
```

核心是这里

```
Tinc.m00 = 1.0f;		Tinc.m10 = step[2];		Tinc.m20 = -step[1];	Tinc.m30 = step[3];Tinc.m01 = -step[2];	Tinc.m11 = 1.0f;		Tinc.m21 = step[0];		Tinc.m31 = step[4];Tinc.m02 = step[1];		Tinc.m12 = -step[0];	Tinc.m22 = 1.0f;		Tinc.m32 = step[5];Tinc.m03 = 0.0f;		Tinc.m13 = 0.0f;		Tinc.m23 = 0.0f;		Tinc.m33 = 1.0f;
```

```
Tinc.m30 = step[3];Tinc.m31 = step[4];Tinc.m32 = step[5];
```

平移向量

```
Tinc.m03 = 0.0f;		Tinc.m13 = 0.0f;		Tinc.m23 = 0.0f;		Tinc.m33 = 1.0f;
```

齐次坐标

```
Tinc.m00 = 1.0f;		Tinc.m10 = step[2];		Tinc.m20 = -step[1];Tinc.m01 = -step[2];	Tinc.m11 = 1.0f;		Tinc.m21 = step[0];	Tinc.m02 = step[1];		Tinc.m12 = -step[0];	Tinc.m22 = 1.0f;
```

旋转矩阵