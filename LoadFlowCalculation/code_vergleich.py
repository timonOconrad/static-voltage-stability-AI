# Imported python packages and files
import multiprocessing as mp
import numpy as np
import time
from newtown_PV_pq_div import newtonrapson
from gaus_PV_pq_div import gausseidel
from powerfactory_API import powerfactory_thread
import pandas as pd


#print("Number of processors: ", mp.cpu_count())

# generation of the cases
def generate_data(leng_y_matrix):

    # generation of the cases for load und generator
    database_G = np.zeros(leng_y_matrix, dtype=complex)
    database_L = np.zeros(leng_y_matrix, dtype=complex)

    # varaition of the data according to the thesis
    for j in range(leng_y_matrix):
            # Set np.nan at the first position
        if j <= 1:
            database_L[j] = 0 + 1j * 0
        else:
            # Generate a complex number with random real and imaginary part
            database_L[j] = np.around(np.random.uniform(0, 99), decimals=0) * 1e6 + np.around(
            np.random.uniform(0, 99), decimals=0) * 1e6 * 1j

        if j == 1:
            database_G[j] = np.around(np.random.uniform(0, 199), decimals=0) * 1e6 +0j
        else:
            database_G[j] = 0 + 1j * 0

        # Alternatively, test values can also be used, these are then to be activated
        """
        P_G = np.array([0, 40, 0, 0, 0]) * 1e6
        Q_G = np.array([0, 0, 0, 0, 0]) * 1e6
        database_G = P_G + 1j * Q_G

        P = np.array([0, 0, -45, -40, -60]) * 1e6
        Q = np.array([0, 0, -15, -5, -10]) * 1e6
        database_L = P + 1j * Q
        """

    return -database_L, database_G


# Start of the Main file
if __name__ == '__main__':
    # Input value for calculation

    # Define the voltages
    u_start = np.array([
        1.06 + 0j,
        1.0 + 0j,
        1.0 + 0j,
        1.0 + 0j,
        1.0 + 0j
    ]) * 230e3

    # Define the bus type
    bus_typ = np.array([
        1, #1 = Slack
        2, #2 =  PV
        3, #3 =  PQ
        3,
        3
    ])

    # Define the admittance matrix
    y_matrix = np.array([[0.01181474-0.03523623j, -0.0094518+0.02835539j, -0.00236295+0.00708885j, 0+0j, 0+0j],
                  [-0.0094518+0.02835539j, 0.02047889-0.06111532j, -0.0031506+0.0094518j, -0.0031506+0.0094518j, -0.0047259+0.01417769j],
                  [-0.00236295+0.00708885j, -0.0031506+0.0094518j, 0.02441714-0.07304342j, -0.01890359+0.05671078j, 0+0j],
                  [0+0j, -0.0031506+0.0094518j, -0.01890359+0.05671078j, 0.02441714-0.07304348j, -0.00236295+0.00708885j],
                  [0+0j, -0.0047259+0.01417769j, 0+0j, -0.00236295+0.00708885j, 0.00708885-0.02111531j]])

    # defien number of passes
    num_runs = 1




    # Create a manager object and a common list, for the multiprocessing
    manager = mp.Manager()

    # Create a pool of processes
    pool = mp.Pool(mp.cpu_count())
    #print(mp.cpu_count())
    if int(num_runs / mp.cpu_count())>1:
        chunksize = int(num_runs / mp.cpu_count())
    else:
        chunksize=1

    # Generate random values based on the YMatrix
    s_values = pool.map(generate_data, [len(y_matrix)] * num_runs, chunksize)


    #print(s_G_values)

    pool.close()
    pool.join()
    # Save results

    async_results_newton = []

    pool = mp.Pool(mp.cpu_count())
    start_newton = time.time()
    i = 0

    # application of the Newton-Raphson method
    for s_multi in s_values:
        s_L_values = s_multi[0]
        s_G_values = s_multi[1]
        i =+ 1
        newtonrapson_results= pool.apply_async(newtonrapson, args=(bus_typ, y_matrix, - s_L_values, s_G_values, u_start))
        async_results_newton.append((newtonrapson_results))


    # End the pool of processes
    pool.close()
    pool.join()


    end_newton = time.time()


    time.sleep(1)
    print("netowng done")

    # start of a new multiprocessing, this may not be necessary, since several calculations are also executed in parallel. here this was only left to stop the time.
    pool = mp.Pool(mp.cpu_count())
    start_gaus = time.time()
    i = 0
    async_results_gaus = []
    # application of the Gaus-Seidel method
    for s_multi in s_values:
        s_L_values = s_multi[0]
        s_G_values = s_multi[1]
        gaus_results = pool.apply_async(gausseidel, args=(bus_typ, y_matrix, s_L_values, s_G_values, u_start))
        async_results_gaus.append((gaus_results))

    # End the pool of processes
    pool.close()
    pool.join()
    end_gaus = time.time()
    time.sleep(1)
    print("gaus done")

    # separating the generated data for powerfactory
    s_L_values = [result[0] for result in s_values]
    s_G_values = [result[1] for result in s_values]
    start_power = time.time()
    powerfactory_results = []

    #execute the non-mutliprocessing capable powerfactory method
    result_powerfactory = powerfactory_thread(s_L_values, s_G_values, u_start, powerfactory_results)
    end_power = time.time()



    print(f"___________________________________")
    print(f"Time taken for Power {num_runs} steps {(end_power - start_power)}  s")
    print(f"Time taken for Newton {num_runs} steps {(end_newton - start_newton)}  s")
    print(f"Time taken for Gaus {num_runs} steps {(end_gaus - start_gaus)}  s")

    # Create empty lists for each method
    newton_data, gaus_data, powerfactory_data = [], [], []

    # Collect data for Newton
    for async_result_newton in async_results_newton:
        try:
            u_newton, evaluation_newton_v, diag_dU_dQ_newton = async_result_newton.get(timeout=1)
            newton_data.append((u_newton, evaluation_newton_v, diag_dU_dQ_newton))
        except Exception as e:
            newton_data.append((None, None, None))

    # Collect data for Gaus
    for async_result_gaus in async_results_gaus:
        try:
            u_gaus, evaluation_gaus, diag_dU_dQ_gaus = async_result_gaus.get(timeout=1)
            gaus_data.append((u_gaus, evaluation_gaus, diag_dU_dQ_gaus))
        except Exception as e:
            gaus_data.append((None, None, None))

    # Collect data for Powerfactory
    for S_G, S_L, U_powerfactory, evaluation, dia_dv_dQ in powerfactory_results:
        powerfactory_data.append((U_powerfactory, evaluation, dia_dv_dQ))

    # Create a combined list for the DataFrame
    combined_data = []
    for n, g, p in zip(newton_data, gaus_data, powerfactory_data):
        combined_data.append({
            "Newton_U": n[0],
            "Newton_Evaluation": n[1],
            "Newton_Diag": n[2],
            "Gaus_U": g[0],
            "Gaus_Evaluation": g[1],
            "Gaus_Diag": g[2],
            "Powerfactory_U": p[0],
            "Powerfactory_Evaluation": p[1],
            "Powerfactory_Diag": p[2]
        })

    # Convert the data into a DataFrame
    df = pd.DataFrame(combined_data)

    # Display the DataFrame
    print(df)


    def extract_evaluation_table(df):
        # Create a new DataFrame with only the evaluation columns
        evaluation_df = df[["Newton_Evaluation", "Gaus_Evaluation", "Powerfactory_Evaluation"]]

        # Optional: rename the columns to make them easier to use
        evaluation_df.columns = ["Newton", "Gaus", "Powerfactory"]

        return evaluation_df

    def extract_dia_table(df):
        # Create a new DataFrame with only the evaluation columns
        evaluation_df = df[["Newton_Diag", "Gaus_Diag", "Powerfactory_Diag"]]

        # Optional: rename the columns to make them easier to use
        evaluation_df.columns = ["Newton", "Gaus", "Powerfactory"]

        return evaluation_df

    # readout of the evaluation:
    evaluation_table = extract_evaluation_table(df)
    print(evaluation_table)


    def count_values_in_evaluation_columns(df):
        for column in ["Newton_Evaluation", "Gaus_Evaluation", "Powerfactory_Evaluation"]:
            print(f"\nValue counting for {column}:")
            print(df[column].value_counts())


    def calculate_evaluation_metrics(evaluation_df):
        # Check if Newton_Evaluation matches Powerfactory_Evaluation
        is_match = evaluation_df["Newton"] == evaluation_df["Powerfactory"]

        # Berechne die Metriken
        true_positive = sum(is_match & (evaluation_df["Newton"] == True))
        false_positive = sum(~is_match & (evaluation_df["Newton"] == True))
        true_negative = sum(is_match & (evaluation_df["Newton"] == False))
        false_negative = sum(~is_match & (evaluation_df["Newton"] == False))

        return {
            "True Positive": true_positive,
            "False Positive": false_positive,
            "True Negative": true_negative,
            "False Negative": false_negative
        }

    def calculate_evaluation_metrics_gaus(evaluation_df):
        # Check if Newton_Evaluation matches Powerfactory_Evaluation
        is_match = evaluation_df["Gaus"] == evaluation_df["Powerfactory"]

        # Calculate the metrics
        true_positive_G = sum(is_match & (evaluation_df["Gaus"] == True))
        false_positive_G = sum(~is_match & (evaluation_df["Gaus"] == True))
        true_negative_G = sum(is_match & (evaluation_df["Gaus"] == False))
        false_negative_G = sum(~is_match & (evaluation_df["Gaus"] == False))

        return {
            "True Positive": true_positive_G,
            "False Positive": false_positive_G,
            "True Negative": true_negative_G,
            "False Negative": false_negative_G
        }


    # Benefit of the function:
    evaluation_metrics = calculate_evaluation_metrics(evaluation_table)
    print("Evaluation Metrics Newton:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    evaluation_metrics_gaus = calculate_evaluation_metrics_gaus(evaluation_table)
    print("Evaluation Metrics Gaus:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    print(extract_dia_table(df))

    import pandas as pd
    import sys
    from time import sleep
    import os

    # Set the output buffer to a larger value
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)


    row_index= 1
    data = df
    row = data.iloc[row_index]
    row_string = row.to_string().encode('utf-8')
    sys.stdout.buffer.write(row_string + b'\n\n')
    sleep(1)
    # Output the number of lines
    print("Total number of rows in the file: ", len(data))