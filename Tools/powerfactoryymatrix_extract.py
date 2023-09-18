import sys
import time
import numpy as np


################### Functions ###################
# Funktion zur Überprüfung, ob ein Wert in der Liste vorkommt
def contains(value, lst):
    for item in lst:
        if value in item:
            return True
    return False


def Z_Trans(Trans_var):
    Trans_name = getattr(Trans_var, 'loc_name')

    U_kom = (getattr(Trans_var, 'm:u1r:bushv') + getattr(Trans_var, 'm:u1i:bushv') * 1j) * Trans_var.GetUnom()

    S_kom = getattr(Trans_var, 'm:Q:bushv') * 1j + getattr(Trans_var, 'm:P:bushv')

    Z = (U_kom ** 2 / S_kom) * getattr(Trans_var,'t:x1pu')  ## hinweis vo nsia wie man die impedanz von transforamtoren bestimmt. x1 von Z_Base
    return Trans_name, Z


def Z_Gen(Gen_var):
    Gen_name = getattr(Gen_var, 'loc_name')

    U_kom = (getattr(Gen_var, 'c:uini:r') + getattr(Gen_var, 'c:uini:r') * 1j) * Gen_var.GetUnom()

    S_kom = getattr(Gen_var, 'm:Q:bus1') * 1j + getattr(Gen_var, 'm:P:bus1')

    try:
        Z = U_kom ** 2 / S_kom
    except:
        Z = 0


    return Gen_name, Z


def Z_Load(Load_var):
    Load_name = getattr(Load_var, 'loc_name')
    U_kom = (getattr(Load_var, 'm:u1r:bus1') + getattr(Load_var, 'm:u1i:bus1') * 1j) * getattr(Load_var,
                                                                                               'm:U1l:bus1')  ## überfürfen welche U #todo
    S_kom = getattr(Load_var, 'm:Q:bus1') * 1j + getattr(Load_var, 'm:P:bus1')

    try:
        U_kom ** 2 / S_kom
    except ZeroDivisionError:
        print("Fehler")
        Z=0
    else:
        Z = U_kom ** 2 / S_kom


    return Load_name, Z


def Z_Line(Line_var):
    Z_name = getattr(Line_var, 'loc_name')
    # Zrad = np.radians(getattr(Line_var, 'phiz1'))
    # Z = getattr(Line_var, 'Z1')*np.exp(1j*Zrad)
    R = getattr(Line_var, 'R1')
    X = getattr(Line_var, 'X1')
    Z = R + X * 1j
    # print(Z_name, " ", 1/Z)
    return Z_name, Z


def Z_Node(Line_var, A):
    Z_name = getattr(Line_var, 'loc_name')
    # Zrad = np.radians(getattr(Line_var, 'phiz1'))
    # Z = getattr(Line_var, 'Z1')*np.exp(1j*Zrad)
    #C1 = getattr(Line_var, 'C1')
    #f = getattr(Line_var, 'c:frnom')
    # Ltg1 = getattr(Line_var, 'dline') schon im C1 enthalten
    B1 = getattr(Line_var, 'B1')/1e6
    #print(C1)
    #print(f)
    print("##################")
    #print(B1)
    #print(A)
    if B1 > 0:
        Z = 1j / B1
    else:
        Z = 0

    # print(Z_name, " ", Z," ", R," ", X )

    return Z_name, Z


def delete_values(matrix, values):
    for row in matrix:
        for col_idx, col in enumerate(row):
            row[col_idx] = [val for val in col if val not in values]
    return matrix

################### Code ###################
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP1\Python\3.9")

if __name__ == "__main__":
    import powerfactory as pf

app = pf.GetApplication()

if app is None:
    raise Exception('getting Powerfactory application failed')
# app.Show()

project = app.ActivateProject("IEEE5-Bus-Masterarbeit")
project = app.GetActiveProject()

load_dict = {}
loads = app.GetCalcRelevantObjects("*.ElmLod")

for i in loads:
    load_dict[i.loc_name] = i

Line_dict = {}
Lines = app.GetCalcRelevantObjects('*.ElmLne')
"""for i in Lines:
    Line_dict[i.loc_name] = i

laenge = 1
Line_dict["Line 1-2"].SetAttribute("dline", laenge)
Line_dict["Line 1-5"].SetAttribute("dline", laenge)
Line_dict["Line 2-3"].SetAttribute("dline", laenge)
Line_dict["Line 2-5"].SetAttribute("dline", laenge)
Line_dict["Line 3-4"].SetAttribute("dline", laenge)
Line_dict["Line 4-5"].SetAttribute("dline", laenge)
"""
# load_value = load_dict["Load A"].GetAttribute("plini")
# print(load_value)

# load_value = 100
# load_dict["Load A"].SetAttribute("plini", load_value)

# load_test = load_dict["Load A"].GetAttribute("plini")
# print(load_test)

Loadflow = app.GetFromStudyCase('ComLdf')  # get load flow object

if Loadflow.Execute() < 1:  # execute load flow

    # get the buses and print their voltage<
    Bus_Data = ["Name", "U1", "Phi in Bogenmaß"]  # check if powerfyctory data in arc dimension
    Buses = app.GetCalcRelevantObjects('*.ElmTerm')
    # print(len(Buses))
    n = int(len(Buses))
    A = np.empty((n, n), dtype=object)

    i = 0
    for bus in Buses:  # loop through list

        Bus_Data = np.vstack([Bus_Data, [getattr(bus, 'loc_name'), getattr(bus, 'm:u1') * bus.GetUnom(),
                                         getattr(bus, 'm:phiu') * np.pi / 180]])
        bustype = bus.GetBusType()
        Cubicles = bus.GetConnectedCubicles()
        bustest = bus

        A[i, i] = []
        for Cubic in Cubicles:
            # print(tefsd.GetFullName())
            connection = getattr(Cubic, 'obj_id')
            value = getattr(connection, 'loc_name')

            A[i, i].append(value)
            # print(value)
            # print(i)

        i = i + 1
        # print(A)

    # Füllung der Matrix A
    for i in range(len(Buses)):
        for j in range(len(Buses)):
            if i != j and A[i][j] is None:  # nur leere Felder betrachten
                common = [x for x in A[i][i] if
                          contains(x, A[j][j])]  # gemeinsame Werte in den Listen von A[i,i] und A[j,j]
                if common:
                    A[i][j] = common
                    A[j][i] = common  # da die Matrix symmetrisch ist, muss auch A[j,i] gesetzt werden
                else:
                    A[i][j] = []
                    A[j][i] = []

    Load_Data = ["Name", "Z1"]
    Loads = app.GetCalcRelevantObjects('*.ElmLod')
    for load in Loads:  # loop through list
        Load_Data = np.vstack([Load_Data, Z_Load(load)])
    # print(Load_Data)
    Gen_Data = ["Name", "Z1"]
    Generators = app.GetCalcRelevantObjects('*.ElmSym')
    for gen in Generators:  # loop through list
        Gen_Data = np.vstack([Gen_Data, Z_Gen(gen)])
    # print(Gen_Data)

    # Löschen der Generator und last Daten aus der Matrix
    # values_to_delete = ['General Load A', 'General Load C', 'General Load B','Synchronous Machine(1)', 'Synchronous Machine']

    # Extrahiere die Namen aus Load_Data und Gen_Data

    # Kombiniere die Namen in einem Array
    values_to_delete = [row[0] for row in Load_Data[1:]] + [row[0] for row in Gen_Data[1:]]
    A = delete_values(A, values_to_delete)

    # Ausgabe der aktualisierten Matrix

    Trans_Data = ["Name", "Z1"]
    Line_Data = ["Name", "Z1"]
    Node_Data = ["Name", "Z1"]

    Transformators = app.GetCalcRelevantObjects('*.ElmTr2')
    for trans in Transformators:  # loop through list
        Trans_Data = np.vstack([Gen_Data, Z_Trans(trans)])

    # get the lines and thier Z
    Lines = app.GetCalcRelevantObjects('*.ElmLne')
    for line in Lines:  # loop through list
        Line_Data = np.vstack([Line_Data, Z_Line(line)])
        Node_Data = np.vstack([Node_Data, Z_Node(line, A)])
    # print(Line_Data)
    # print(Node_Data)
    diag_A = np.diag(A)
    # Gib die aktualisierte Matrix A aus
    # print(diag_A)

    result_matrix = [[0 for _ in row] for row in diag_A]

    # Fülle die Matrix mit Werten aus Node_Data
    for i, row in enumerate(diag_A):
        for j, col in enumerate(row):
            for node in Node_Data:
                if col == node[0]:
                    result_matrix[i][j] += complex(node[1])

    # Berechne die Umkehrung jedes Elements
    # überprüfen ob zulässig bei divission durch null
    for i, row in enumerate(result_matrix):
        for j, col in enumerate(row):
            if col == 0:
                result_matrix[i][j] = 0
            else:
                result_matrix[i][j] = 1 / col

    # Berechne die Summe jeder Zeile
    row_sums = [sum(row) for row in result_matrix]
    # print(row_sums)

    # print(Line_Data)
    # print(Trans_Data)
    # print(Gen_Data)
    # print(Load_Data)

    # Konvertieren Sie alle Eingabe-Arrays in NumPy-Arrays
    Line_Data = np.array(Line_Data)
    print(Line_Data)
    print(Node_Data)
    Trans_Data = np.array(Trans_Data)
    # Gen_Data = np.array(Gen_Data)
    # Load_Data = np.array(Load_Data)

    # print(A)
    print("########################")
    # Überprüfen Sie die Anzahl von Dimensionen jedes Arrays
    arrays = []
    if Line_Data.ndim > 1 and Line_Data.shape[0] > 1:
        arrays.append(Line_Data[1:])
    if Trans_Data.ndim > 1 and Trans_Data.shape[0] > 1:
        arrays.append(Trans_Data[1:])
    # if Gen_Data.ndim > 1 and Gen_Data.shape[0] > 1:
    #    arrays.append(Gen_Data[1:])
    # if Load_Data.ndim > 1 and Load_Data.shape[0] > 1:
    #    arrays.append(Load_Data[1:])

    # Verketten Sie die Arrays
    all_data = np.concatenate(arrays, axis=0)
    data_dict = {row[0]: complex(row[1]) for row in all_data}

    # Berechnen Sie die Matrix
    Admittanz_Matrix = np.empty(A.shape, dtype=complex)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            elements = A[i, j]
            if len(elements) == 0:
                Admittanz_Matrix[i, j] = 0
            else:
                diagonal = i == j
                factor = 1 if diagonal else -1
                total = sum([factor * 1 / data_dict[element] for element in elements if element in data_dict])
                Admittanz_Matrix[i, j] = total

    Admittanz_Matrix = Admittanz_Matrix - np.diag(row_sums)

    U = np.empty((1, 6))
    for i in range(1, 6):
        #print(float(Bus_Data[i, 1]) * np.exp(1j * float(Bus_Data[i, 2])))
        print(abs(float(Bus_Data[i, 1]) * np.exp(1j * float(Bus_Data[i, 2]))))

    print(Admittanz_Matrix)


    print("########################")


else:
    print("no convergence in load flow")


