import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	state_dict ={}
	obj_dict = {}
	for t,i in zip(tags,range(len(tags))):
		state_dict[t]=i
	S = len(state_dict)
	pi = np.zeros(S)
	A = np.zeros([S,S])
	B = np.array([])
	for line in train_data:
		line.words = sentance_normalizing(line.words)
		pi[state_dict[line.tags[0]]] +=1.0
		pi[state_dict[line.tags[0]]] +=1.0
		if line.words[0] not in obj_dict:
			obj_dict[line.words[0]] = len(obj_dict.keys())
			if B.size == 0:
				B = np.append(B,np.zeros(S)).reshape([-1,1])
			else:
				B = np.append(B,np.zeros(S).reshape([-1,1]),axis=1)
		B[state_dict[line.tags[0]],obj_dict[line.words[0]]] +=1.0
		for prev_tag,tag,word in zip(line.tags[:-1],line.tags[1:],line.words[1:]):
			A[state_dict[prev_tag],state_dict[tag]] +=1.0 
			if word not in obj_dict:
				obj_dict[word] = len(obj_dict.keys())
				B = np.append(B,np.zeros(S).reshape([-1,1]),axis=1)
			B[state_dict[tag],obj_dict[word]] +=1.0
	pi = pi/ np.sum(pi)
	A = A/ np.sum(A,axis=1)
	A[np.isnan(A)] = 0.0
	B = B/ np.sum(B,axis=0)
	B[np.isnan(B)] = 0.0
	np.maximum(B,10**-6,out=B)
	model = HMM(pi,A,B,obj_dict,state_dict)
	###################################################
	# Edit here
	###################################################
	return model


# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	###################################################
	S = len(model.state_dict)
	for sentance in test_data:
		s = sentance_normalizing(sentance.words)
		for word in s:
			if word not in model.obs_dict:
				model.obs_dict[word] = len(model.obs_dict.keys())
				model.B = np.append(model.B,np.full((S), 10**-6).reshape([-1,1]),axis=1)
		tagging.append(model.viterbi(s))

	return tagging
	
def sentance_normalizing(sentance):
	ret_Val = []
	for word in sentance:
		ret_Val.append(word.lower())
	return ret_Val