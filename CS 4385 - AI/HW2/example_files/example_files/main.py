import sys # Gets the command line_num arguments
import operator # to perform functions to given valid operators
import re # to replace occurrences of a particular sub-string with another sub-string
import copy # to perform deep copy


# operator.gt gets the greater value
# operator.lt gets the less than value
# operator.eq gets the equals to(==) value
# operator.ne gets the 'not equals to' value

valid_operators = {">": operator.gt,"<": operator.lt,"=": operator.eq,"!": operator.ne}
counter = 0
forward_checking = True


class Variable():
    VR = None
    DM = None
    AS = None

def main():
    global counter
    variables_collection = {}

    with open(sys.argv[1], errors='ignore') as filename:
        for index, line_num in enumerate(filename):
            line_num = re.sub(r'\n', '', line_num)
            line_num = re.sub(r'[ \t]+$', '', line_num)
            
            variable = Variable() # variable class object instantiation
            variable.VR = line_num[0]
            sample_DOM = []
            
            for line in line_num[3:].split(' '):
                sample_DOM.append(int(line))
            
            variable.DM = sample_DOM
            variable.AS = None
            variables_collection[variable.VR] = variable

    constraints_collection = []
    with open(sys.argv[2], errors='ignore') as filename:
        for index, line_num in enumerate(filename):
            line_num = re.sub(r'\n', '', line_num)
            line_num = re.sub(r'[ \t]+$', '', line_num)
            constraints_collection.append((line_num[0], line_num[2], line_num[4]))

    # change this to a function based on arg3 later
    
    if sys.argv[3] == "none":
        #print(type(sys.argv[3]))
        forward_checking = False
    else:
        forward_checking = True

    results = recurBacktracking({}, variables_collection, constraints_collection, forward_checking)
    if results is not False:
        iterator = 0
        counter += 1
        print(counter, ". ", end="", sep="")
        for value in results.keys():
            if iterator is len(results.keys()) - 1:
                print(value, "=", results[value], " solution", sep="")
            else:
                print(value, "=", results[value], ", ",sep="",end="")
            iterator += 1


def recurBacktracking(assigned, variables_collection, constraints_collection, forward_checking):
    global counter
    
    if all(variable.AS != None for variable in variables_collection.values()):
        return assigned

    var = selectUnassignedVariable(variables_collection, constraints_collection)

    orderedDM = sortDM(variables_collection, constraints_collection, var)
    
    for vals in orderedDM:
        for val in vals:
            val = int(val)
            flag = True
            for cons in constraints_collection:
                if cons[0] is variables_collection[var].VR:
                    if variables_collection[cons[2]].AS is None:
                        continue
                    else:
                        flag = valid_operators[cons[1]](val, int(variables_collection[cons[2]].AS))
                        
                elif cons[2] is variables_collection[var].VR:
                    if variables_collection[cons[0]].AS is None:
                        continue
                    else:
                        flag = valid_operators[cons[1]](int(variables_collection[cons[0]].AS), val)
                        
                if flag is False:
                    c = 0
                    counter += 1
                    print(counter, ". ", end="", sep="")
                    for i in assigned.keys():
                        if c is len(assigned.keys()) - 1:
                            print(i, "=", assigned[i], ", ",sep="",end="")
                            print(variables_collection[var].VR, "=", val, " failure", sep="")
                        else:
                            print(i, "=", assigned[i], ", ",sep="",end="")
                        c += 1
                    if counter >= 30:
                        SystemExit
                    break

            if flag is True:
                variables_collection[var].AS = val
                assigned[var] = val
                resultvariables_collection = None
                #forward checking begins
                if forward_checking is True:
                    
                    resultvariables_collection = forward_checking_function(copy.deepcopy(variables_collection), constraints_collection, var)
                    # Avoid DMs's Emptieness
                    for variable in resultvariables_collection.values():
                        if len(variable.DM) == 0:
                            c = 0
                            counter += 1
                            print(counter, ". ", end="", sep="")
                            for i in assigned.keys():
                                if c is len(assigned.keys()) - 1:
                                    print(variables_collection[var].VR, "=", val, " failure", sep="")
                                else:
                                    print(i, "=", assigned[i], ", ",sep="",end="")
                                c += 1
                            if counter >= 30:
                                SystemExit
                            continue
                else:
                    resultvariables_collection = variables_collection

                result = recurBacktracking(assigned, resultvariables_collection, constraints_collection, forward_checking)
                if result is not False:
                    return result
                variables_collection[var].AS = None
                assigned.pop(var)
    
    return False


def forward_checking_function(variables_collection, constraints_collection, var):
    assignedValue = variables_collection[var].AS
   
    for cons in constraints_collection:
        if cons[0] is variables_collection[var].VR and variables_collection[cons[2]].AS is None:
                removalList = []
                for value in variables_collection[cons[2]].DM:
                    if valid_operators[cons[1]](assignedValue, value) != True:
                        removalList.append(value)
                
                for r in removalList:
                    variables_collection[cons[2]].DM.remove(r)
   
        if cons[2] is variables_collection[var].VR and variables_collection[cons[0]].AS is None:
                removalList = []
                for value in variables_collection[cons[0]].DM:
                    if valid_operators[cons[1]](value, assignedValue) != True:
                        removalList.append(value)
                for r in removalList:
                    variables_collection[cons[0]].DM.remove(r)
                        
    return variables_collection


def selectUnassignedVariable(variables, constraints_collection):
    var = None
    varList = []
    for v in variables.keys():
    
        if variables[v].AS == None:
          
            if var == None:
                var = v
                varList.append(v)
          
            elif len(variables[var].DM) > len(variables[v].DM):
                var = v
                varList = [v]
          
            elif len(variables[var].DM) == len(variables[v].DM):
                varcount = 0
                variablecount = 0
              
                varcount += sum(
                    1 for i in constraints_collection if i[0] == variables[var].VR and variables[i[2]].AS == None)
               
                varcount += sum(
                    1 for i in constraints_collection if variables[i[0]].AS == None and i[2] == variables[var].VR)
              
                variablecount += sum(
                    1 for i in constraints_collection if i[0] == variables[v].VR and variables[i[2]].AS == None)
              
                variablecount += sum(
                    1 for i in constraints_collection if variables[i[0]].AS == None and i[2] == variables[v].VR)
              
                if varcount < variablecount:
                    var = v
                    varList = [v]
                elif varcount == variablecount:
                    varList.append(v)
    
    return var


def sortDM(variables_collection, constraints_collection, var):

    value_constraints = {}
    for value in variables_collection[var].DM:
        value = int(value)
        temp_value = 0
        for con in constraints_collection:
            
            if con[0] is variables_collection[var].VR and variables_collection[con[2]].AS is None:
                for compValue in variables_collection[con[2]].DM:
                    if not valid_operators[con[1]](value, int(compValue)):
                        temp_value += 1
            elif variables_collection[con[0]].AS is None and con[2] == variables_collection[var].VR:
                for compValue in variables_collection[con[0]].DM:
                    if not valid_operators[con[1]](int(compValue), value):
                        temp_value += 1
        
        if temp_value in value_constraints:
            value_constraints[temp_value].append(int(value))
        else:
            value_constraints[temp_value] = [int(value)]

    ordered_DM = []
    for new_entry in sorted(value_constraints.keys()):
        ordered_DM.append(value_constraints[new_entry])

    return ordered_DM


if __name__ == "__main__":
    main()