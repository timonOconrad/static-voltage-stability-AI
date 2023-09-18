# Imported python packages
import sys
import numpy as np
import time

# Link to the API, important, select correct version
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP1\Python\3.9")

import powerfactory as pf


# powerfacotry calculation
def powerfactory_case(s_multi,s_G_multi, u_start, app):
    # app.Show() # anzeigen der oberfläche

    # select the prepared variant
    project = app.ActivateProject("IEEE5-Bus-Masterarbeit_PVwithoutload")
    project = app.GetActiveProject()

    # convert to MW
    S_L= s_multi/ 1e6
    S_G = s_G_multi/1e6

    # convert to p.u.
    U = u_start / 230e3

    # create the generator dictionary
    gen_dict = {}
    gens = app.GetCalcRelevantObjects("*.ElmSym")


    for i in gens:
        gen_dict[i.loc_name] = i
    # changing the values for the PV generator
    gen_dict["Gen_PV_Bus"].SetAttribute("pgini", np.real(S_G[1]))
    gen_dict["Gen_PV_Bus"].SetAttribute("qgini", np.imag(S_G[1]))
    gen_dict["Gen_PV_Bus"].SetAttribute("usetp", np.real(U[1]))

    gen_dict["Slack"].SetAttribute("usetp", np.real(U[0]))

    # create the generator dictionary
    load_dict = {}
    loads = app.GetCalcRelevantObjects("*.ElmLod")

    for i in loads:
        load_dict[i.loc_name] = i

    """
    load_dict["Load A"].SetAttribute("plini", np.real(S_L[1]) * -1)
    load_dict["Load A"].SetAttribute("qlini", np.imag(S_L[1]) * -1)
    load_dict["Load A"].SetAttribute("u0", np.real(U[1]))
    """
    # changing the values for the PQ- Loads

    load_dict["Load B"].SetAttribute("plini", np.real(S_L[2]) * -1)
    load_dict["Load B"].SetAttribute("qlini", np.imag(S_L[2]) * -1)
    load_dict["Load B"].SetAttribute("u0", np.real(U[2]))

    load_dict["Load C"].SetAttribute("plini", np.real(S_L[3]) * -1)
    load_dict["Load C"].SetAttribute("qlini", np.imag(S_L[3]) * -1)
    load_dict["Load C"].SetAttribute("u0", np.real(U[3]))

    load_dict["Load D"].SetAttribute("plini", np.real(S_L[4]) * -1)
    load_dict["Load D"].SetAttribute("qlini", np.imag(S_L[4]) * -1)
    load_dict["Load D"].SetAttribute("u0", np.real(U[4]))



    Buses = app.GetCalcRelevantObjects('*.ElmTerm')
    n = int(len(Buses))
    Data = np.zeros((n, n), dtype=object)
    U_powerfactory = np.zeros((n), dtype=complex)

    # Load flow calculation
    Loadflow = app.GetFromStudyCase('ComLdf')

    if Loadflow.Execute() < 1:  # execute load flow
        S_G[1] = gen_dict["Gen_PV_Bus"].GetAttribute("m:P:bus1")+1j*gen_dict["Gen_PV_Bus"].GetAttribute("m:Q:bus1")
        S_G[0] = gen_dict["Slack"].GetAttribute("m:P:bus1") + 1j * gen_dict["Slack"].GetAttribute("m:Q:bus1")

        i = 0
        for bus in Buses:  # loop through list für the buses
            # U_powerfactory[i] = getattr(bus, 'm:u1')*bus.GetUnom()* np.exp(1j*(getattr(bus, 'm:phiu')*np.pi/180))*1e3
            Data[i, 0] = getattr(bus, 'loc_name')
            Data[i, 1] = getattr(bus, 'm:u1') * bus.GetUnom() * np.exp(
                1j * (getattr(bus, 'm:phiu') * np.pi / 180)) * 1e3
            i = i + 1

        # Sensitivity analysis
        com = app.GetFromStudyCase('*.ComVstab')
        n = int(len(Buses))
        dv_dQ_raw = np.empty((n, n), dtype=object)
        dv_dQ_name = np.empty((n, n), dtype=object)
        i = 0
        for flexbus in Buses:
            app.PrintInfo(flexbus)
            com.c_butldf = Loadflow
            com.p_bus = flexbus
            j=0
            if com.Execute() < 1:
                for bus in Buses:  # loop through list
                    dv_dQ_raw[i,j]=getattr(bus, 'm:dvdQ')
                    dv_dQ_name[i, j]=getattr(bus, 'loc_name')+getattr(flexbus, 'loc_name')
                    j = j + 1
                    if getattr(flexbus, 'loc_name') == getattr(bus, 'loc_name'):
                        Data[i, 2] = getattr(bus, 'm:dvdQ') #/ np.sqrt(3) * bus.GetUnom() / 1e3
                        # print(getattr(flexbus, 'loc_name'), getattr(bus, 'loc_name'))
                        # g= getattr(bus, 'GetUnom')
                        # print(g)
            else:
                print("no sensitiv")
                evaluation = 0
            i = i + 1
        #print("#############dv_dQ############")
        #print(dv_dQ_raw)
        #print(dv_dQ_name)



        #print("#############dv_dQ############")
        """dv_dQ_raw= dv_dQ_raw[:, :-1]
        dv_dQ_raw = dv_dQ_raw[:-1, :]

        dia_dv_dQ = np.diag(dv_dQ_raw)
        new_dia_dv_dQ = np.copy(dia_dv_dQ)
        new_dia_dv_dQ[1], new_dia_dv_dQ[2] = dia_dv_dQ[2], dia_dv_dQ[1]
        """
        # print(dv_dQ_raw)
        # print(dv_dQ_name)

        Data = sorted(Data, key=lambda row: row[0])
        #print(Data)
        dia_dv_dQ = [row[2] for row in Data] ### anpassung wegen pv  ### wichtig

        U_powerfactory = [row[1] for row in Data]

        dv_dQ=dv_dQ_raw.astype('float64')

        """
        # Determinante von J berechnen
        #print(np.linalg.det(J_R))
        # ganz komisch, hier vermischt powerfactory die busknoten
        if np.linalg.det(J_R) > 0:
            evaluation = 1
        else:
            evaluation = 0
            #print(dia_dv_dQ)
        """
        #try:
        #print(dv_dQ)

        # Evaluation
        dia_dv_dQ = np.diag(dv_dQ)
        if np.all(dia_dv_dQ >= 0):
             evaluation = 1
        else:
             evaluation = 0
        #print(evaluation)
        """except ZeroDivisionError:
            evaluation = 0
            print("Warnung, Teilung durch 0")
        """
    else:
        print("no convergence in load flow")
        evaluation = 0
        dia_dv_dQ = np.zeros(n, dtype=object)
        dia_dv_dQ[:] = np.nan


    #print("evaluation power", evaluation)
    #print([abs(value) for value in U_powerfactory])
    return S_G,S_L, U_powerfactory, evaluation, dia_dv_dQ

# powerfacotry case treatment

def powerfactory_thread(s_values, s_G_values, u_start, results_powerfactory):
    app = pf.GetApplicationExt()
    #s_GaL_values = []
    start = time.time()
    if app is None:
        raise Exception('getting Powerfactory application failed')

    #s_GaL_values.append((s_values, s_G_values))
    duration=[]
    i=0
    reset=0
    for s_G, s_L  in zip(s_G_values, s_values):
        start_step = time.time()
        result_powerfactory = powerfactory_case(s_L, s_G, u_start, app)
        results_powerfactory.append(result_powerfactory)
        i=i+1
        end_step = time.time()
        step= end_step - start_step
        duration.append(end_step-start_step)
        reset = reset + 1
        if reset ==75:
            reset =0
            print(f"Progress {np.round((i / len(s_G_values) * 100),2)} %, Duration so far  {np.round((end_step-start)/60,5)} min, Duration per step {np.round(step*1000)} ms, Remaining time {np.round((step * (len(s_G_values) - i))/60,5)} min, Total {np.round(((step * (len(s_G_values) - i))+(end_step-start))/60,5)} min")

    # Exit the PowerFactory program. The API cannot be called again
    del app
    end = time.time()
    print(f"Mean{(end-start) / len(s_G_values)} s, Max {max(duration)} s, Min {(min(duration))} s")