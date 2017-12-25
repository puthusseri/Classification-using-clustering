import re
import csv
import random
import math
import operator


from flask import Flask, render_template, json, request
app = Flask(__name__)
@app.route("/")
def main():
        return render_template('index.html')
@app.route("/home")
def main1():
        return render_template('index.html')
@app.route("/about")
def abt():
        return render_template('aboutus.html')
@app.route("/diseasePrediction")
def diseasePre():
        return render_template('diseasePrediction.html')
@app.route("/heartDiseasePrediction")
def heartDiseasePre():
        return render_template('heartDisease.html')

@app.route("/diseasePredictionSubmit",methods=['POST'])
def usingPatternMatching():
        '''These are the symptoms entered by user in the text box'''

        #s=" i am having fever and headache. also severe stomachache."
        #s="my eyes are comming out due to severe eyepain . yesterday i have headache also"
        #s="with high stomachache and headache. also a small amount of swetting"

        s = request.form['s']
        s=s.lower()
        dnam = request.form['dname']

        stopwords=["i","are","am","but","that","my","and","to","have","also"]
        sympt=['fever','headache','stomachache',"eyepain","swetting"]

        '''Removimg stopwords from the sentence.'''
        for i in stopwords:
            pattern=r'\b'+i+r'\b'
            s=re.sub(pattern,'',s)

        '''Splitting the string into words'''
        l=[]
        s=s.split()
        for i in s:
            l.append([i,0])

        '''Changeing to dictionary'''
        di=dict(l)
        orr={}
        a=[]

        '''Finding keywords'''
        for i in di.iterkeys():
            for j in sympt:
                if i==j:
                    orr[i]=di[i]

        '''Changeing the keyword into a list '''
        for i in orr.iterkeys():
            a.append(i)

        orr=a


        '''Our disease database'''
        do={'flue':['fever','headache','stomachache'],'myophia':['headache','eyepain'],'hungry':['headache','stomachache','swetting']}

        '''For identifying which disease have max sympt'''
        disease={'flue':0,'myophia':0,'hungry':0}

        '''Counting the no of symptoms in every disease in database'''
        for i in do.iterkeys():
            for j in do[i]:
                for k in orr:
                    if k==j:
                        disease[i]=disease[i]+1
        m=0
        mk=" "

        '''Finding the disease with max number of sympt matched.'''
        for i in disease.iterkeys():
            if m<disease[i]:
                m=disease[i]
                mk=i

        print "Disease is "+mk
        if mk==" ":
                mk="No disease"
        return render_template('diseasePrediction.html',resultantDisease=mk,rname=dnam)





@app.route('/heartDisease',methods=['POST'])
def indexHeart():
        testset=[]
        r = request.form['nam']
        #Converting the datas into a uniform type
        age = float(int(request.form['age']))
        sex = float(request.form['sex'])
        chestPain = float(request.form['chestPain'])
        restingBP = float(request.form['restingBP'])
        cholestoral = float(request.form['cholestoral'])
        sugar = float(request.form['sugar'])
        electroCardiographic = float(request.form['electroCardiographic'])
        maximumHeartRate = float(request.form['maximumHeartRate'])
        angina = float(request.form['angina'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        majorVessels = float(request.form['majorVessels'])
        thal = float(request.form['thal'])
        testset.extend((age ,sex ,chestPain ,restingBP ,cholestoral ,sugar ,electroCardiographic ,maximumHeartRate ,angina ,oldpeak ,slope ,majorVessels ,thal))

        #algorithm=request.form['algorithm']

        nreturnValue=m(testset)

        kreturnValue=n(testset)

        creturnValue=o(testset)

        if nreturnValue[1]==1.0:
                string1 = 'Heart Disease'
        else:
                print "return value="

                string1 = 'Not Heart Disease'
        if kreturnValue[1]==1.0:
                string2 = 'Heart Disease'
        else:
                print "return value="

                string2 = 'Not Heart Disease'
        if creturnValue[1]==1.0:
                string3 = 'Heart Disease'
        else:
                print "return value="

                string3 = 'Not Heart Disease'

        return render_template('heartDisease.html',rname=r,knnaccuracy=kreturnValue[0],caccuracy=creturnValue[0],naccuracy=nreturnValue[0],knnresult=string2,nresult=string1,cresult=string3)


def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
        try:
                return sum(numbers)/float(len(numbers))
        except TypeError:
                numbers=list(numbers)

                #numbers=[int(i) for i in numbers]
                s=0
                s=sum(numbers)
                return s/len(numbers)


def stdev(numbers):
        numbers=[float(i) for i in numbers]
        avg = mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):

        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.iteritems():
                probabilities[classValue] = 1
                for i in range(len(classSummaries)):
                        mean, stdev = classSummaries[i]
                        try:
                                x = inputVector[i]
                        except IndexError:
                                pass
                        probabilities[classValue] *= calculateProbability(x, mean, stdev)
        return probabilities



def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]



def m(testset):
        returnValue=[]
        filename = 'h.csv'
        splitRatio = 0.67
        dataset = loadCsv(filename)
        trainingSet, testSet = splitDataset(dataset, splitRatio)
        print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
        # prepare model
        summaries = summarizeByClass(trainingSet)
        # test model
        predictions = getPredictions(summaries, testSet)
        accuracy = getAccuracy(testSet, predictions)
        result = predict(summaries, testset)
        returnValue.append(accuracy)
        returnValue.append(result)
        print('Accuracy: {0}%').format(accuracy)
        return returnValue




def o(testset):
        returnValue=[]
        #clustered data
        filename = 'a.csv'
        a=loadCsv(filename)




        #main data
        filename = 'h.csv'
        lines = csv.reader(open(filename, "rb"))
        data = list(lines)
        for i in range(len(data)):
            data[i] = [float(x) for x in data[i]]
            data[i].append(a[i][1]-1)

        s = separateByClass(data)

        count=True
        for key,values in s.items():
            if count:
                a=values
                count=False
            b=values

        trainingSet1, testSet1 = splitDataset(a, 0.67)
        trainingSet, testSet = splitDataset(b, 0.67)
        trainingSet.extend(trainingSet1)
        testSet.extend(testSet1)
        summaries = summarizeByClass(trainingSet)
        # test model
        predictions = getPredictions(summaries, testSet)
        accuracy = getAccuracy(testSet, predictions)
        result = predict(summaries, testset)
        returnValue.append(accuracy)
        returnValue.append(result)
        print('Accuracy:with clustering {0}%').format(accuracy)
        return returnValue

def n(testset):
        # prepare data
	returnValue=[]
	filename = 'h.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)

	accuracy = getAccuracy(testSet, predictions)
	neighbors = getNeighbors(dataset, testset, k)
	result = getResponse(neighbors)
	returnValue.append(accuracy)
	returnValue.append(result)
	print('Accuracy: {0}%').format(accuracy)
	return returnValue

if __name__ == "__main__":
        app.run()
