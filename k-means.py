from numpy import *



# dataSet������,k �صĸ���
# disMeas�������ȣ�Ĭ��Ϊŷ����þ���
# createCent,��ʼ���ѡȡ
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  # ������
    clusterAssment = mat(zeros((m, 2)))  # m*2�ľ���
    centroids = createCent(dataSet, k)  # ��ʼ��k������
    clusterChanged = True
    while clusterChanged:  # �����಻�ٱ仯
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):  # �ҵ����������
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # ��1��Ϊ�������ģ���2��Ϊ����
            clusterAssment[i, :] = minIndex, minDist ** 2
        print (centroids)

        # ��������λ��
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

  for cent in range(k):
      ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
      centroids[cent,:] = mean(ptsInClust, axis=0)


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

def draw(data,center):
    length=len(center)
    fig=plt.figure
    # ����ԭʼ���ݵ�ɢ��ͼ
    plt.scatter(data[:,0],data[:,1],s=25,alpha=0.4)
    # ���ƴص����ĵ�
    for i in range(length):
        plt.annotate('center',xy=(center[i,0],center[i,1]),xytext=\
        (center[i,0]+1,center[i,1]+1),arrowprops=dict(facecolor='red'))
    plt.show()


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    # �����һ��Ϊ��𣬵ڶ���ΪSSE
    clusterAssment = mat(zeros((m, 2)))
    # ����һ�����ǵ�����
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # create a list with one centroid
    for j in range(m):  # ����ֻ��һ�����ǵ����
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2

    # ���Ĵ���
    while (len(centList) < k):
        lowestSSE = inf
        # ����ÿһ�����ģ����ԵĽ��л���
        for i in range(len(centList)):
            # �õ����ڸ����ĵ�����
            ptsInCurrCluster =\ dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # �Ը����Ļ��ֳ�����
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # ����ôػ��ֺ��SSE
            sseSplit = sum(splitClustAss[:, 1])
            # û�в��뻮�ֵĴص�SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print
            "sseSplit, and notSplit: ", sseSplit, sseNotSplit
            # Ѱ����С��SSE���л���
            # ������һ���ؽ��л��ֺ�SSE��С
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # �������Ĳ���
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print
        'the bestCentToSplit is: ', bestCentToSplit
        print
        'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
        :] = bestClustAss  # reassign new clusters, and SSE
    return mat(centList), clusterAssment