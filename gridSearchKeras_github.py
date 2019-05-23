#performs grid search of hyperparameters

import re
from itertools import product
import settings
import dnn_keras_github
import tensorflow as tf
import math
import numpy as np

def main():	
	tf.logging.set_verbosity(tf.logging.INFO)
	variables = settings.args_keys
	hidden_units=[[1000,  1000, 1000, 1]]
	cv = [5]
	stratified = [True]
	learning_rate = [0.0003]
	dropout = [0.1]
	normalization = [True]
	pca=[False]
	pcaVec = [1000]
	batch_size = [64]
	train_steps = [2500]
	reorg_norm_factor = [27211.4] 
	train_percent = [0.80]
	
	
	big = 0.01 #learning rate upper end
	little = 0.0001 #learning rate lower end
	high = math.log10(big) 
	low = math.log10(little) 
	
	gridList = list(product(hidden_units,cv,stratified,learning_rate,
		dropout,normalization,pca,pcaVec,batch_size,train_steps, reorg_norm_factor,train_percent))
	
	'''
	for i in range(len(gridList)):
		# log scale search for learning rate
		grid = list(gridList[i])
		grid[3] = math.pow(10,np.random.rand()*(high-low) + low)
		gridList[i] = tuple(grid)
		# log scale search for learning rate
	'''
	oldGrid = gridList[0]

	for grid in gridList:
		print(len(grid))		
		if oldGrid == gridList[0]:
			for i in range(len(grid)):
				print('i=',i)
				settings.update_setting(settings.path, "Settings", list(settings.args_keys)[i], grid[i])			
		else:
			for i in range(len(grid)):
				if not oldGrid[i]  == grid[i] :
					print(list(settings.args_keys)[i],grid[i])
					settings.update_setting(settings.path, "Settings", list(settings.args_keys)[i], grid[i])
		
		print('*** Done modifiying settings.ini ***') 
		
		dnn_keras_github.main()
		
		oldGrid = grid
	
if __name__ == "__main__":
	main()
