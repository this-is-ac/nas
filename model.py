def decodeUnit(unit, x, y, operands) :
	if unit[0] < len(operators) :
		operand1 = operators[unit[0]](x, y)
	else :
		num = unit[0] - len(operators)
		operand1 = 	operands[num + 1]

	if unit[2] < len(operators) :
		operand2 = operators[unit[2]](x, y)
	else :
		num = unit[2] - len(operators)
		operand2 = 	operands[num + 1]

	return 	binaryOps[unit[4]](unaryOps[unit[1]](operand1), unaryOps[unit[3]](operand2))

def decode(individual, x, y):
	operands = [1]
	for i in range(numUnits):
		operands.append(decodeUnit(individual[i*5:(i+1)*5], x, y, operands))
	return norm1(operands[-1], 1)

def CustomKernelGramMatrix(X1, X2, individual, l1, l2) :
	gram_matrix = decode(individual, X1, X2)
	gram_matrix = np.reshape(gram_matrix, [l1, l2])
	return gram_matrix	

def model_fn(actions, cust_train_data):
    #op1_l1, op2_l1, u1_l1, u2_l1, b_l1, op1_l2, op2_l2, u1_l2, u2_l2, b_l2 = actions
    train_train_x, train_train_y = cust_train_data
    k_matrix = CustomKernelGramMatrix(train_train_x, train_train_y, actions, len(train_train_x), len(train_train_x))
    return k_matrix
