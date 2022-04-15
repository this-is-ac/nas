import numpy as np

def norm1(x, k) :
	if np.array(x).shape[-1] == 1 :
		return x
	return np.linalg.norm(x, ord = k, axis = -1)

def decodeUnit(unit, x, y, operands) :
    return 	unit[4](unit[1](unit[0](x,y)), unit[3](unit[2](x,y)))

def decode(individual, x, y, numUnits):
	operands = [1]
	for i in range(numUnits):
		operands.append(decodeUnit(individual[i*5:(i+1)*5], x, y, operands))
	return norm1(operands[-1], 1)

def CustomKernelGramMatrix(X1, X2, individual, l1, l2, numUnits) :
	gram_matrix = decode(individual, X1, X2, numUnits)
	gram_matrix = np.reshape(gram_matrix, [l1, l2])
	return gram_matrix	

def model_fn(actions, cust_train_data, numUnits):
    #op1_l1, u1_l1, op2_l1, u2_l1, b_l1, op1_l2, u1_l2, op2_l2, u2_l2, b_l2 = actions
    
    train_train_x, train_train_y, validation_train_x, validation_train_y = cust_train_data
    train_k_matrix = CustomKernelGramMatrix(train_train_x, train_train_y, actions, int(np.sqrt(len(train_train_x))), int(np.sqrt(len(train_train_x))), numUnits)
    val_train_matrix = CustomKernelGramMatrix(validation_train_x, validation_train_y, actions, int(np.sqrt(len(train_train_x))), int(np.sqrt(len(train_train_x))), numUnits)
    
    return train_k_matrix,val_train_matrix
