import tushare as ts
import pandas as pd
import numpy as np
import time
import psutil
import pandas as pd
import numpy as np
#import quandl # 获取股票数据
from datetime import date
import matplotlib.pyplot as plt
# TOKEN='dc0fc631cf596141f871d54540a5daec939bed3da73b81bca23889d5'
# pro = ts.pro_api(token=TOKEN)
# data = pro.daily(ts_code ='600109.SH',\
# start_date = '20220101',end_date='20220601')
# data.to_csv('41.csv')
f=open("C:\\Users\\zhengjiang\\Desktop\\2.txt",encoding='utf-8')
#print(f)
lines=f.readline()
data=[]
while lines:
	lines1=lines.split()
	if lines1 != []:
		lines1.remove(lines1[0])
		data.append(lines1)
		# print(len(lines1))
		# print(lines1[1])
		# print(lines1[0])
		# print(lines1)
		lines=f.readline()
f.close()
#print(data)

class treeNode:
	def __init__(self,nameValue,numOccur,parentNode):
		self.name=nameValue
		self.count=numOccur
		self.nodelink=None
		self.parent=parentNode
		self.children={}
	def inc(self,numOccur):
		self.count+=numOccur
	def disp(self,ind=1):
		print(' '*ind,self.name,'',self.count)
		for child in self.children.values():
			child.disp(ind+1)
def loadSimpDat():
	simDat = data
	return simDat
def createInitSet(dataSet):
	retDict={}
	for trans in dataSet:
		key=frozenset(trans)
		if retDict.get(key,0)>0:
			retDict[frozenset(trans)]+=1
		else:
			retDict[frozenset(trans)]=1
	return retDict
def updateHeader(nodeToTest, targetNode):
	while nodeToTest.nodelink != None:
		nodeToTest = nodeToTest.nodelink
	nodeToTest.nodelink = targetNode
def updateFPtree(items,inTree,headerTable,count):
	#items 是需要的事务,intree 是根节点，count 事务数量
	if items[0] in inTree.children:#出现过的，在同一点，不需要另外开辟
		inTree.children[items[0]].inc(count)
	else:
		inTree.children[items[0]]=treeNode(items[0],count,inTree)
	if headerTable[items[0]][1]==None:
		headerTable[items[0]][1]=inTree.children[items[0]]
	else:#链表跟新
		updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
if len(items) > 1:
	updateFPtree(items[1::],inTree.children[items[0]],headerTable,count)
def createFPtree(dataset,minsup=74):
	headerTable={}
	freqItemSet=set()
	for trans in dataset:
		for item in trans:
			headerTable[item]=headerTable.get(item,0)+dataset[trans]
	for k in headerTable.keys():
		if headerTable[k]<minsup:
			pass
		else:
			freqItemSet.add(k)
	if len(freqItemSet)==0:
		return None,None
	for k in headerTable:
		headerTable[k]=[headerTable[k],None]#一个数量，一个指针
	retTree=treeNode('Null Set',1,None)
	for tranSet,count in dataset.items():
		localD={}
		for item in tranSet:
			if item in freqItemSet:
				localD[item]=headerTable[item][0]
		if len(localD)>0:
			#按照支持度对数列进行从大到小排序,sorted 固定写法
			#这里注意，如果 p[0]是字母，上面一句，是字符化的整数，就用下面一句。
			#orderedItem = [v[0] for v in sorted(localD.items(), key=lambda p:(p[1], -ord(p[0])), reverse=True)]
			orderedItem = [v[0] for v in sorted(localD.items(), key=lambda p:(p[1], int(p[0])), reverse=True)]
			#添加到一颗树上

			updateFPtree(orderedItem,retTree,headerTable,count)
	return retTree, headerTable
def ascendFPtree(leafNode, prefixPath):
	if leafNode.parent != None:
		prefixPath.append(leafNode.name)
		ascendFPtree(leafNode.parent, prefixPath)
def findPrefixPath(base,myheadertable):
	treenode=myheadertable[base][1]#指向的第一个节点
	cond={}
	while treenode!=None:
		preFixPath=[]
		ascendFPtree(treenode,preFixPath)
		if len(preFixPath)>1:
			cond[frozenset(preFixPath[1:])]=treenode.count
		treenode=treenode.nodelink
	return cond
def mineFPtree(inTree,headerTable,minSup,preFix,freqItemList):
	freq={}
	for k in headerTable.keys():
		if headerTable[k][0]<minSup:
			pass
		else:
			freq[k]=headerTable[k][0]
			#是字符化的整数，就用下面一句，字母就用上面一句。
	#bigL=[v[0] for v in sorted(freq.items(), key=lambda p:(p[1], ord(p[0])))]
	bigL=[v[0] for v in sorted(freq.items(), key=lambda p:(p[1], int(p[0])))]
	#从最后一项最小项开始
	for base in bigL:
		newFreqSet=preFix.copy()
		newFreqSet.add(base)
		#只要是在头表中的，肯定是满足要求的
		freqItemList.append(newFreqSet)
		zishujvji=findPrefixPath(base,headerTable)
		zishu,zitoubiao=createFPtree(zishujvji,minSup)
		if zitoubiao!=None:
			mineFPtree(zishu,zitoubiao,minSup,newFreqSet,freqItemList)
simpDat = loadSimpDat()
initSet = createInitSet(simpDat)
myFPtree, myHeaderTab = createFPtree(initSet,74)
freqItems = []
mineFPtree(myFPtree, myHeaderTab,74, set([]), freqItems)
#print(len(freqItems))
#print(freqItems)
total=[]
for item in freqItems:
	if len(item) == 3:
		total.append(item)
print(total)
#print(len(total))
datazuhe1=pd.read_csv("C:\\Users\\zhengjiang\\Desktop\\1.csv",parse_dates=['trade_date'], index_col='trade_date')
#二、投资组合的收益计算
# 设置组合权重，存储为 numpy 数组类型
#portfolio_weights = np.array([0.3, 0.3, 0.4])
# 将收益率数据拷贝到新的变量 stock_return 中，这是为了后续调用的方便
stock_return = datazuhe1.copy()
print(stock_return)
# 计算加权的股票收益
#WeightedReturns = stock_return.mul(portfolio_weights, axis=1)
# 计算投资组合的收益
#datazuhe1['Portfolio'] = WeightedReturns.sum(axis=1)
# 绘制组合收益随时间变化的图
# datazuhe1.Portfolio.plot()
# plt.show()
# 累积收益曲线绘制函数
def cumulative_returns_plot(name_list):
	for name in name_list:
		CumulativeReturns = ((1+datazuhe1[name]).cumprod()-1)
		CumulativeReturns.plot(label=name)
	plt.legend()
	plt.show()
# 设置投资组合中股票的数目
numstocks = 3
# 平均分配每一项的权重
#portfolio_weights_ew = np.repeat(1/numstocks, numstocks)
# 计算等权重组合的收益
#datazuhe1['Portfolio_EW'] = stock_return.mul(portfolio_weights_ew, axis=1)
# .sum(axis=1)
# 绘制累积收益曲线
#cumulative_returns_plot(['Portfolio', 'Portfolio_EW'])
#三、相关性和协方差
# 计算相关矩阵
correlation_matrix = stock_return.corr()
# 输出相关矩阵
#print(correlation_matrix)
# 导入 seaborn
import seaborn as sns
# 创建热图
sns.heatmap(correlation_matrix,
			annot=True,
			cmap="YlGnBu",
			linewidths=0.3,
			annot_kws={"size": 8})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
#plt.show()
# 计算协方差矩阵
cov_mat = stock_return.cov()
#print(cov_mat)
# 年化协方差矩阵
cov_mat_annual = cov_mat * 252
# 输出协方差矩阵
#print(cov_mat_annual)
#四寻找最优投资组合
# 设置模拟的次数
#蒙特卡洛模拟 Markowitz 模型
number = 10000
# 设置空的 numpy 数组，用于存储每次模拟得到的权重、收益率和标准差
random_p = np.empty((number, 5))
# 设置随机数种子，这里是为了结果可重复
np.random.seed(123)
# 循环模拟 10000 次随机的投资组合
for i in range(number):
	# 生成 3 个随机数，并归一化，得到一组随机的权重数据
	random3 = np.random.random(3)
	random_weight = random3 / np.sum(random3)
	#print(random_weight)
	# 计算年化平均收益率
	mean_return = stock_return.mul(random_weight, axis=1).sum(axis=1).mean()
	annual_return = (1 + mean_return)**252 - 1
	# 计算年化的标准差，也称为波动率
	random_volatility = np.sqrt(np.dot(random_weight.T,
	np.dot(cov_mat_annual,random_weight)))
	# 将上面生成的权重，和计算得到的收益率、标准差存入数组 random_p 中
	random_p[i][:3] = random_weight
	random_p[i][3] = annual_return
	random_p[i][4] = random_volatility
# 将 numpy 数组转化成 DataFrame 数据框
RandomPortfolios = pd.DataFrame(random_p)
ticker_list=["600109","601901","601688"]
# 设置数据框 RandomPortfolios 每一列的名称
RandomPortfolios.columns = [ticker + "_weight" for ticker in ticker_list] + ['Returns', 'Volatility']
#print(RandomPortfolios)
# 绘制散点图
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
plt.show()
#风险最小组合
# 找到标准差最小数据的索引值
min_index = RandomPortfolios.Volatility.idxmin()
# 在收益-风险散点图中突出风险最小的点
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[min_index,'Volatility']
y = RandomPortfolios.loc[min_index,'Returns']
plt.scatter(x, y, color='red')
plt.show()
# 提取最小波动组合对应的权重, 并转换成 Numpy 数组
GMV_weights = np.array(RandomPortfolios.iloc[min_index, 0:numstocks])
print(GMV_weights)
# 计算 GMV 投资组合收益
datazuhe1['Portfolio_GMV'] = stock_return.mul(GMV_weights, axis=1).sum(axis=1)
# 绘制累积收益曲线
#cumulative_returns_plot(['Portfolio_GMV'])
#夏普最优组合
# 设置无风险回报率为 0
risk_free = 0
# 计算每项资产的夏普比率
RandomPortfolios['Sharpe'] = (RandomPortfolios.Returns - risk_free) / RandomPortfolios.Volatility
print(RandomPortfolios['Sharpe'])
#绘制收益-标准差的散点图，并用颜色描绘夏普比率
plt.scatter(RandomPortfolios.Volatility, RandomPortfolios.Returns, c=RandomPortfolios.Sharpe)
plt.colorbar(label='Sharpe Ratio')
plt.show()
# 找到夏普比率最大数据对应的索引值
max_index = RandomPortfolios.Sharpe.idxmax()
# 在收益-风险散点图中突出夏普比率最大的点
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[max_index,'Volatility']
y = RandomPortfolios.loc[max_index,'Returns']
print(x)
print(y)
plt.scatter(x, y, color='red')
plt.show()
# 提取最大夏普比率组合对应的权重，并转化为 numpy 数组
MSR_weights = np.array(RandomPortfolios.iloc[max_index, 0:numstocks])
print(MSR_weights)
# 计算 MSR 组合的收益
datazuhehuice1=pd.read_csv("C:\\Users\\zhengjiang\\Desktop\\huice1.csv",parse_dates=['trade_date'], index_col='trade_date')
stock_return_huice1=datazuhehuice1.copy()
datazuhehuice1['Portfolio_MSR'] = stock_return_huice1.mul(MSR_weights, axis=1).sum(axis=1)
datazuhehuice1['Portfolio_GMV'] = stock_return_huice1.mul(GMV_weights, axis=1).sum(axis=1)
# 累积收益曲线绘制函数
def cumulative_returns_plothuice(name_list):
	for name in name_list:
		CumulativeReturns = ((1+datazuhehuice1[name]).cumprod()-1)
		CumulativeReturns.plot(label=name)
	plt.legend()
	plt.show()
# 绘制累积收益曲线
cumulative_returns_plothuice(['Portfolio_MSR'])
cumulative_returns_plothuice(['Portfolio_GMV'])
datazuhehuice1['Portfolio'] = stock_return_huice1.mul(MSR_weights,axis=1).sum(axis=1)
datazuhehuice1.Portfolio.plot()
plt.show()
# 计算投资组合的标准差
portfolio_volatility1 = np.sqrt(np.dot(MSR_weights.T,  np.dot(cov_mat_annual, MSR_weights)))
portfolio_volatility2 = np.sqrt(np.dot(GMV_weights.T, np.dot(cov_mat_annual, GMV_weights)))
print(portfolio_volatility1)
print(portfolio_volatility2)
