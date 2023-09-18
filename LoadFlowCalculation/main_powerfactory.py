import multiprocessing as mp
import numpy as np
import time
from powerfactory_API import powerfactory_thread
import pandas as pd
import os

#print("Number of processors: ", mp.cpu_count())

###  Basis ###
# Path to the parquet file
file_path = 'data/data_IEEE5_powerfactory_99_199.parquet'


def add_data_to_parquet(file_path, new_data):
    # Check if the file already exists
    file_exists = os.path.isfile(file_path)

    # Load data or create empty DataFrame
    data = pd.DataFrame() if not file_exists else pd.read_parquet(file_path)

    # Check if the file is larger than 500 MB
    if file_exists and os.path.getsize(file_path) > 1000 * 1024 * 1024:  # Dateigröße in Bytes
        confirm = input("The file exceeds 1000 MB. Do you want to use the storage space (Y/N)? ")
        if confirm != "Y":
            print("Storage space usage rejected.")
            return
    print(os.path.getsize(file_path) / 1024 / 1024, "MB")
    # Daten mit vorhandenen Daten zusammenführen

    new_data['u_powerfactory_real'] = new_data['u_powerfactory'].apply(lambda x: np.real(x))
    new_data['u_powerfactory_imag'] = new_data['u_powerfactory'].apply(lambda x: np.imag(x))
    new_data.drop('u_powerfactory', axis=1, inplace=True)


    # Convert 'S' column to 1-dimensional arrays
    new_data['P_G'] = new_data['S_G'].apply(lambda x: np.real(x))
    new_data['Q_G'] = new_data['S_G'].apply(lambda x: np.imag(x))
    new_data.drop('S_G', axis=1, inplace=True)

    new_data['P_L'] = new_data['S_L'].apply(lambda x: np.real(x))
    new_data['Q_L'] = new_data['S_L'].apply(lambda x: np.imag(x))
    new_data.drop('S_L', axis=1, inplace=True)

    new_data.to_parquet(file_path, engine='pyarrow')

    data = pd.concat([data, new_data], ignore_index=True)

    # Drop 'S' column

    data.to_parquet(file_path, engine='pyarrow')

    # Übersicht anzeigen
    num_rows = len(data)

    print(f"Number of lines: {num_rows}")
    count_zeros = data['evaluation_powerfactory'].value_counts().get(0, 0)
    print(f"Frequency of 0 in % powerfactory': {count_zeros / num_rows}")
#### Fuktions ####
def generate_data(leng_y_matrix):

    database_G = np.zeros(leng_y_matrix, dtype=complex)
    database_L = np.zeros(leng_y_matrix, dtype=complex)

    for j in range(leng_y_matrix):
            # Set np.nan at the first position
        if j == 0:
           database_L[j] = 0 + 1j * 0
        else:
            # Generate a complex number with random real and imaginary part
            database_L[j] = np.around(np.random.uniform(0, 99), decimals=0) * 1e6 + np.around(
            np.random.uniform(0, 99), decimals=0) * 1e6 * 1j

        if j == 1:
            database_G[j] = np.around(np.random.uniform(0, 199), decimals=0) * 1e6 +0j
        else:
            database_G[j] = 0 + 1j * 0


    """
    P_G = np.array([0, 40, 0, 0, 0]) * 1e6
    Q_G = np.array([0, 0, 0, 0, 0]) * 1e6
    database_G = P_G + 1j * Q_G

    P = np.array([0, 0, -45, -40, -60]) * 1e6
    Q = np.array([0, 0, -15, -5, -10]) * 1e6
    database_L = P + 1j * Q
    """
    return -database_L, database_G

start = time.time()


if __name__ == '__main__':
    # Eingabewert für die Datenstruktur
    u_start = np.array([
        1.06 + 0j,
        1.0 + 0j,
        1.0 + 0j,
        1.0 + 0j,
        1.0 + 0j
    ]) * 230e3

    y_matrix = np.array(
        [[0.01181474 - 0.03523623j, -0.0094518 + 0.02835539j, -0.00236295 + 0.00708885j, 0 + 0j, 0 + 0j],
         [-0.0094518 + 0.02835539j, 0.02047889 - 0.06111532j, -0.0031506 + 0.0094518j, -0.0031506 + 0.0094518j,
          -0.0047259 + 0.01417769j],
         [-0.00236295 + 0.00708885j, -0.0031506 + 0.0094518j, 0.02441714 - 0.07304342j, -0.01890359 + 0.05671078j,
          0 + 0j],
         [0 + 0j, -0.0031506 + 0.0094518j, -0.01890359 + 0.05671078j, 0.02441714 - 0.07304348j,
          -0.00236295 + 0.00708885j],
         [0 + 0j, -0.0047259 + 0.01417769j, 0 + 0j, -0.00236295 + 0.00708885j, 0.00708885 - 0.02111531j]])

    # Number of passes
    num_runs = 1000

    # Creating a Manager object and a shared list
    manager = mp.Manager()

    powerfactory_results = manager.list()

    # Create a pool of processes
    pool = mp.Pool(mp.cpu_count())
    #print(mp.cpu_count(), "Kerne")
    print("-------------------------------------------")
    if int(num_runs / mp.cpu_count())>1:
        chunksize = int(num_runs / mp.cpu_count())
    else:
        chunksize=1

    # Generate random values based on the YMatrix
    generation_S = pool.map(generate_data, [len(y_matrix)] * num_runs, chunksize)
    t1= time.time()


    # Extract the s_L_values and s_G_values from the results
    s_L_values = [result[0] for result in generation_S]
    s_G_values = [result[1] for result in generation_S]

    #print(s_L_values)
    #print(s_G_values)

    t2= time.time()
    print(f"Generation {(t2 - start)}  s")

    # Save results
    result_powerfactory = mp.Process(target=powerfactory_thread, args=(s_L_values, s_G_values, u_start, powerfactory_results))
    result_powerfactory.start()

    # End the pool of processes
    pool.close()
    pool.join()

    # End the result_powerfactory process
    result_powerfactory.join()

    # Iterate over the result_list to extract the results and store them in result_data

    result_data = []

    for s_G_values, s_L_values, u_powerfactory, evaluation_powerfactory, dU_dQ_powerfactory in powerfactory_results:
        result_data.append({
            'S_L': s_L_values,
            'S_G': s_G_values,
            'u_powerfactory': u_powerfactory,
            'evaluation_powerfactory': evaluation_powerfactory,
            'diag(dU_dQ)_powerfactory': dU_dQ_powerfactory
        })


    #print(result_data)

    new_data = pd.DataFrame(result_data)

    # Add the data
    add_data_to_parquet(file_path, new_data)

    #print(powerfactory_results)

end = time.time()

#print(f"Time taken is {(end - start)}  s")
