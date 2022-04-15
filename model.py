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
    train_train_x, train_train_y = cust_train_data
    k_matrix = CustomKernelGramMatrix(train_train_x, train_train_y, actions, len(train_train_x), len(train_train_x), numUnits)
    return k_matrix
