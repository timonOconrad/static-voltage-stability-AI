# Imported python packages
import numpy as np

# defining the Jacobian matrix
def JacobianMatrix3p(Y_admittance, U):
    N, M = Y_admittance.shape

    # dP/dU designated as J2
    J2 = np.zeros((N, M))

    for n in range(N):  # rows; one row is omitted - one node is omitted because it is a compensation node (Slack).
        for m in range(M):  # columns one row is omitted- one node is omitted because it is a compensation node (slack).
            if m == n:  # for the diagonal components
                J2[n, m] = 2 * np.abs(U[n]) * np.abs(Y_admittance[n, m]) * np.cos(np.angle(Y_admittance[n, m]))

                for k in range(M):
                    if k != n:
                        J2[n, m] = J2[n, m] + np.abs(Y_admittance[n, k]) * np.abs(U[k]) * np.cos(np.angle(U[n]) - np.angle(U[k]) - np.angle(Y_admittance[n, k]))

            if m != n:  # for undiagonal components
                J2[n, m] = np.abs(Y_admittance[n, m]) * np.abs(U[n]) * np.cos(np.angle(U[n]) - np.angle(U[m]) - np.angle(Y_admittance[n, m]))

    # dP/dphi designated as J1
    J1 = np.zeros((N, M))

    for n in range(N):
        for m in range(M):
            if m == n: # for the diagonal components
                for k in range(M):
                    if k != n:
                        J1[n, m] = J1[n, m] - np.abs(Y_admittance[n, k]) * np.abs(U[k]) * np.abs(U[n]) * np.sin(np.angle(U[n]) - np.angle(U[k]) - np.angle(Y_admittance[n, k]))

            if n != m:  # for undiagonal components
                J1[n, m] = np.abs(Y_admittance[n, m]) * np.abs(U[m]) * np.abs(U[n]) * np.sin(np.angle(U[n]) - np.angle(U[m]) - np.angle(Y_admittance[n, m]))

    # dQ/dU designated as J4
    J4 = np.zeros((N, M))

    for n in range(N):
        for m in range(M): # for the diagonal components
            if m == n:
                J4[n, m] = -2 * np.abs(U[m]) * np.abs(Y_admittance[n, m]) * np.sin(np.angle(Y_admittance[n, m]))

                for k in range(M):
                    if k != n:
                        J4[n, m] = J4[n, m] + np.abs(Y_admittance[n, k]) * np.abs(U[k]) * np.sin(np.angle(U[n]) - np.angle(U[k]) - np.angle(Y_admittance[n, k]))

            if m != n:  # for undiagonal components
                J4[n, m] = np.abs(Y_admittance[n, m]) * np.abs(U[n]) * np.sin(np.angle(U[n]) - np.angle(U[m]) - np.angle(Y_admittance[n, m]))


    # dQ/dphi designated as J3
    J3 = np.zeros((N, M))

    for n in range(N):
        for m in range(M):
            if m == n:
                for k in range(M):
                    if k != n:
                        J3[n, m] = J3[n, m] + np.abs(Y_admittance[n, k]) * np.abs(U[k]) * np.abs(U[n]) * np.cos(
                            np.angle(U[n]) - np.angle(U[k]) - np.angle(Y_admittance[n, k]))

            if m != n:  # for undiagonal components
                J3[n, m] = -np.abs(Y_admittance[n, m]) * np.abs(U[m]) * np.abs(U[n]) * np.cos(
                    np.angle(U[n]) - np.angle(U[m]) - np.angle(Y_admittance[n, m]))

    #J = np.block([[J1[1:N, 1:M], J2[1:N, 1:M]], [J3[1:N, 1:M], J4[1:N, 1:M]]])  # ganze Jacobi - Matrix
    J = np.block([[J1, J2],[J3, J4]])  # whole Jacobi - Matrix

    #Return of the jacobi matrices
    return J, J1, J2, J3, J4

# Definition of the Gaus -Seidel- Mehode
def gausseidel(bus_typ, Y_system, s_L, s_G, U):
    U_memory = []
    evaluation = 0
    try:

        # Adding the apparent power. Since this follows the active sign convention, it can be added instead of subtracted.
        S = s_G + s_L

        N, M = Y_system.shape


        dU_dQ = np.zeros((M - 1, M - 1))
        evaluation = 0
        convergenz= False
        limit_until_converged=200

        PV_indices = np.where(bus_typ == 2)[0]
        #print("PV indes",PV_indices )

        # Remove Slack

        Q_PV = np.zeros(N)

        # Calculations to the limit
        for j in range(limit_until_converged):
            #print(j)

            # Calculation of the PV bus
            if np.any(bus_typ == 2):
                for p in PV_indices:
                    # print(p)
                    hilfsvektor_Q_PV = np.zeros(N, dtype=complex)
                    hilfsvektor_Q_PV_test = np.zeros(N, dtype=complex)
                    for q in range(0, N):
                        hilfsvektor_Q_PV[q]= U[q] * Y_system[p,q]

                        hilfsvektor_Q_PV_test[q] = (abs(U[p]) * abs(U[q]) * abs(Y_system[p, q]) * np.sin(
                            np.angle(Y_system[p, q]) + np.angle(U[q]) - np.angle(U[p])))

                    #print("1: ", -1 * sum(hilfsvektor_Q_PV_test))

                    # Determine the Q for the PV bus
                    Q_PV[p]= -1*np.imag(np.conj(U[p])*sum(hilfsvektor_Q_PV))
                    #print("2: ",Q_PV[p])

            #Q_PV= Q_PV



            #print(S)
            for i in PV_indices:
                S[i] = S[i].real + Q_PV[i] * 1j

            U_save=U[PV_indices]

            # Calculation according to PQ bus
            for m in range(M):
                hilf_vektor = np.zeros(N, dtype=complex)
                for n in range(N):
                    if n == m:
                        hilf_vektor[n] = np.conj(S[n]) / (Y_system[n, n] * np.conj(U[n]))
                    if n != m:
                        hilf_vektor[n] = -Y_system[m, n] / Y_system[m, m] * U[n]
                if m > 0:
                    U[m] = np.sum(hilf_vektor)


            i = 0
            for p in PV_indices:
                U[p] = np.abs(U_save[i]) * np.exp(np.angle(U[p]) * 1j)
                i = i + 1

            #print("2", abs(U))
            U_memory.append(U.copy())

            # early abort condition
            if len(U_memory) >= 3:
                if max(abs(U_memory[-1]- U_memory[-2]))<0.0005:
                    if max(abs(U_memory[-2]- U_memory[-3]))<0.0005:
                        convergenz=True
                        #print("convergenz_gaus", convergenz)
                        break
        # final calculated voltage
        U_end = [value for value in U_memory[-1:]]



        if convergenz:

            # determine the Jacobian matrix early Gaus seidel, for the dV/dQ evaluation
            J, J1, J2, J3, J4 = JacobianMatrix3p(Y_system, U_end[0])
            """J1 = J1[1:, 1:]
            J2 = J2[1:, 1:]
            J3 = J3[1:, 1:]
            J4 = J4[1:, 1:]"""
            #dU_dQ = np.linalg.inv(J4 - np.dot(J3, np.dot(np.linalg.inv(J1), J2)))
            dU_dQ = np.linalg.inv(J4 - np.dot(J3, np.dot(np.linalg.inv(J1), J2)))
            """dU_dQ = np.zeros((M - 1, M - 1))
            for i in range(1, M):
                for j in range(1, M):
                    if (J2[i, j] * J3[i, j] - J1[i, j] * J4[i, j]) != 0:
                        dU_dQ[i - 1, j - 1] = (J1[i, j]) / (J1[i, j] * J4[i, j] - J2[i, j] * J3[i, j])
                    else:
                        dU_dQ[i - 1, j - 1] = np.nan
                        #print("Fehler bei Newton")
            """
            #print(dU_dQ)
            if (np.diag(dU_dQ) >= 0).all():
                evaluation = 1
            else:
                evaluation = 0
            pass

        #print("evaluation Gaus", evaluation)
    except Exception as e:
        print(f"Error in gausseidel: {str(e)}")


    print("Gaus", U_memory[-1])
    return U_memory[-1], evaluation, np.diag(dU_dQ)
