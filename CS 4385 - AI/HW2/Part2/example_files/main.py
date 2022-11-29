import copy
from hashlib import new
import operator
import sys          # Read from sys
import re           # Read from file
from Variable import VARIABLE

counter_global = 0

# Desmotrate CSP

def backTrack(list_a, constraits ,variable,flag_isFC):
    count = 0               # Check to stop
    global counter_global   # Global call
    operators = {"=":operator.eq, ">":operator.gt,"<":operator.lt}      # Check operator from file

    # Stop when all Variable was assigned then sto
    for var in variable.values():
        if var.AS != None:
            count = count + 1
    if count == len(variable):
        return list_a

    # Get new variable then get it domain
    var_Check = selectUnassignValue(constraits,variable)
    new_Domain = sortDomain(constraits,variable, var_Check)

    # Check and work
    for i_new_Domain in new_Domain:
        for ind in i_new_Domain:
            flag = True
            ind = int(ind)

            for index in constraits:
                if index[0] == variable[var_Check].VR and variable[index[2]].AS != None: 
                    flag = operators[index[1]](ind,int(variable[index[2]].AS)) 
                if index[2] == variable[var_Check].VR and variable[index[0]].AS != None: 
                    flag = operators[index[1]](int(variable[index[0]].AS),ind)
                # Display answer
                if flag != True:
                    temp = 0
                    counter_global = counter_global + 1
                    n = len(list_a.keys()) - 1
                    displatCouterGlobal(counter_global)
                    for value in list_a.keys():
                        if temp != n:
                            display (list_a,value)
                        else:
                            displayFalse(list_a,value,variable,var_Check,ind)
                        temp = temp + 1
                    break

            if flag == True:
                variable[var_Check].AS = ind
                list_a[var_Check]=ind
                result_list = None
                # Not use FC (none)
                if flag_isFC != True:
                    result_list = variable

                # Use FC
                else:                 
                    result_list = forward_checking(constraits,copy.deepcopy(variable),var_Check)
                    for re_var in result_list.values():
                        if len(re_var.DM) ==0:
                            index =0
                            counter_global = counter_global +1
                            n = len(list_a.keys()) - 1
                            displatCouterGlobal(counter_global)
                            for value in list_a.keys():
                                if index != n:
                                    displayFalse(list_a,value,variable,var_Check,ind)
                                else:
                                   display (list_a,value)        
                                index = index + 1
                answer = backTrack(list_a,constraits,result_list,flag_isFC)
                if answer is not False:
                    return answer
                variable[var_Check].AS = None
                list_a.pop(var_Check)
    return False

# Display functions
def display (list,value):
    print(value," = ",list[value]," , ",end="", sep="")
# Dispaly false
def displayFalse(list,value,variable,var_Check,ind):
    display(list,value)
    print(variable[var_Check].VR, " = ", ind, " Failure", sep="")
# Dispaly Solution
def displaySolution(list,value):
    print(value," = ",list[value]," Solution.", sep="")
# Display Couter Global
def displatCouterGlobal(counter_global):
    print(counter_global,": ", end="", sep="")

#When using FC this function will work
def forward_checking(constraits ,variable, var_Check):
    operators = {"=":operator.eq, ">":operator.gt,"<":operator.lt}
    assignVar = variable[var_Check].AS

    for index in constraits:
        # Remove
        if index[0] == variable[var_Check].VR and variable[index[2]].AS == None:
            remove_list = []
            for var in variable[index[2]].DM:
                if operators[index[1]](assignVar, var) == False:
                    remove_list.append(var)
            for temp in remove_list:
                variable[index[2]].DM.remove(temp)

        if index[2] == variable[var_Check].VR and variable[index[0]].AS == None:
            remove_list = []
            for var in variable[index[2]].DM:
                if operators[index[1]](var, assignVar) != True:
                    remove_list.append(var)
            for temp in remove_list:
                variable[index[0]].DM.remove(temp)  
        

    return variable


# sort domain
def sortDomain(constraits ,variable, var_Check):
    operators = {"=":operator.eq, ">":operator.gt,"<":operator.lt}
    constr = {}
    for value in variable[var_Check].DM:
        value = int(value)
        temp = 0
        for index in constraits:
            
            if index[0]==variable[var_Check].VR and variable[index[2]].AS is None:
                for index_value in variable[index[2]].DM:
                    if not operators[index[1]](value,int(index_value)):
                        temp = temp + 1
            if index[2]==variable[var_Check].VR and variable[index[0]].AS is None:
                for index_value in variable[index[0]].DM:
                    if not operators[index[1]](int(index_value),value):
                        temp = temp + 1
        if temp in constr:
            constr[temp].append(int(value))
        else:
            constr[temp] = [int(value)]

    # Return new Domain with sorted
    new_Domain = []
    for num in sorted(constr.keys()):
        new_Domain.append(constr[num])

    return new_Domain

# select unassign value
def selectUnassignValue(constraits ,variable):
    selected = None
    selected_list = []
    for var in variable.keys():
        if variable[var].AS == None:
            if selected == None:
                selected = var
                selected_list.append(selected)
            # Find a new value use count and index
            if len(variable[var].DM)==len(variable[selected].DM):
                count = 0
                index = 0

                count = count + sum(1 for con_index in constraits if con_index[0] == variable[selected].VR and variable[con_index[2]].AS == None)
                count = count + sum(1 for con_index in constraits if con_index[2] == variable[selected].VR and variable[con_index[0]].AS == None)
                index = index + sum(1 for con_index in constraits if con_index[0] == variable[var].VR and variable[con_index[2]].AS == None)
                index = index + sum(1 for con_index in constraits if con_index[2] == variable[var].VR and variable[con_index[0]].AS == None)
            if len(variable[var].DM) < len(variable[selected].DM):
                selected = var
                selected_list = [selected]
            if count < index:
                selected = var
                selected_list = [selected]
            if count == index:
                selected_list.append(var)    
                
    return selected


def main():

    global counter_global
    
    # Read .con
    contr = []       # Constraits
    with open(sys.argv[1]) as f_name:
        for i,line in enumerate(f_name):
            line = re.sub(r'\n','',line)
            line = re.sub(r'[ \t]+$','',line)
            line = line.split(' ')
            contr.append(line)
            # print(contr)

    # Read .var
    variables = {}
    with open(sys.argv[2]) as f_name:
        for i,line in enumerate(f_name):
            line = re.sub(r'\n','',line)
            line = re.sub(r'[ \t]+$','',line)

            # Get Variable from file
            label = VARIABLE()
            label.VR = line[0]
            
            # Get domain
            temp = []
            for num in line[3:].split(' '):
                temp.append(int(num))
            label.DM = temp
            variables[label.VR]=label

    flag_isFC = False        # Flage for forward checking
    
    # Set flag for FC
    if sys.argv[3] == "fc":
        flag_isFC = True
    else: 
        flag_isFC = False

    # Get answer
    answer = backTrack({},contr,variables,flag_isFC)
    
    if answer != True:
        counter_global = counter_global + 1
        index = 0
        n = len(answer.keys()) - 1
        displatCouterGlobal(counter_global)
        for value in answer.keys():
            if index != n:
                display (answer,value)
            else:
                displaySolution (answer,value)
            index = index +1


    
if __name__=="__main__":
    main()