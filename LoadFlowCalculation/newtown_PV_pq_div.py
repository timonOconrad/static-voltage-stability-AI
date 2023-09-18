# Imported python packages
import numpy as np

# Definition of the calculation of the voltage
def VoltageCalculation3p(bus_typ, JMatrix, Y_admittance, U, P, Q):
    N, M = Y_admittance.shape

    I = Y_admittance @ U  # Calculated currents in nodes
    Ss = np.diag(U) @ np.conj(I)  # Apparent powers in knots

    PQs = np.concatenate((np.real(Ss), np.imag(Ss)))  # Division of the vector between active and reactive powers

    # Slack recognize
    Slack_indices = np.where(bus_typ == 1)[0]

    #RemoveSlack
    if np.any(bus_typ == 1):
        # Delete digits in the other array
        PQs = np.delete(PQs, Slack_indices+N)
        PQs = np.delete(PQs, Slack_indices)
        P = np.delete(P, Slack_indices)
        Q = np.delete(Q , Slack_indices)
        JMatrix = np.delete(JMatrix, Slack_indices + N, axis=1)
        JMatrix = np.delete(JMatrix, Slack_indices + N, axis=0)
        JMatrix = np.delete(JMatrix, Slack_indices, axis=1)
        JMatrix = np.delete(JMatrix, Slack_indices, axis=0)


    #PQs_Test = np.concatenate((np.real(Ss[1:N]), np.imag(Ss[1:N])))
    #PV Bus detect
    PV_indices = np.where(bus_typ == 2)[0]

    #Q_save = Q[indices]
    #print(Q_save)
    # Check if the number 2 is present in the list
    if np.any(bus_typ == 2):
        # Delete digits in the other array

        PQs = np.delete(PQs, PV_indices-len(Slack_indices)+N-len(Slack_indices))
        #print(PV_indices+N-len(Slack_indices))
        JMatrix = np.delete(JMatrix, PV_indices + N-len(Slack_indices), axis=1)
        JMatrix = np.delete(JMatrix, PV_indices + N-len(Slack_indices), axis=0)

        Q = np.delete(Q, PV_indices-len(Slack_indices))



    #print(PQs)

    delta = (np.concatenate((P, Q)) - PQs)  # Difference between the given powers and calculated powers - interpreted as a step of the algorithm.

    #print(delta)
    try:
        np.linalg.inv(JMatrix)
    except np.linalg.LinAlgError:
        #deltaU = np.full(((N-1)*2), 1) # diskutieren
        #print(deltaU)
        print("Fehler")
        #return None
    else:
        deltaU = np.linalg.inv(JMatrix) @ delta
        #print(deltaU)
    # Formula 3.31

    deltaU = np.insert(deltaU, PV_indices+ N-2, 0)
    for k in range(N-1):
        U[k+1] = (np.abs(U[k+1]) + deltaU[k+N-1]) * np.exp(1j * np.angle(U[k+1]) + 1j * deltaU[k])
        test=deltaU[k+N-1]
        # change to script from Prof. Dzienis, Taylor series

    #Ss=np.insert(deltaU, indices, Q_save)
    Us = U.copy()
    S = Ss.copy()

    #print(abs(Us))
    return Us, S  # Calculation of the new voltages

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

# Parameter für die Lastflussberechnung
def newtonrapson(bus_typ, Y_system, s_L, s_G, U):
    try:
        S = s_G - s_L
        P = np.real(S)
        Q = np.imag(S)

        N, M = Y_system.shape

        dU_dQ = np.zeros((M - 1, M - 1))
        #dU_dQ_test_2 = np.zeros((M - 1, M - 1))
        #dU_dQ_T = np.zeros((M - 1, M - 1))


        # Jacobian matrix method
        u_jacobi = np.empty([5, ], dtype=complex)
        det_jacobi = []
        convergenz = False
        evaluation = 0

        for iter in range(100):  # Step procedure
            J, J1, J2, J3, J4 = JacobianMatrix3p(Y_system, U)
           # print("U",U)

            # determine the new values
            Us, S1 = VoltageCalculation3p(bus_typ, J, Y_system, U, P, Q)
            U = Us
            #print(abs(U))
            det_jacobi.append(np.linalg.det(J))
            u_jacobi = np.vstack((u_jacobi, U))

            # early abort condition
            if len(u_jacobi) >= 3:
                if max(abs(u_jacobi[-1]- u_jacobi[-2]))<0.0005:
                    if max(abs(u_jacobi[-2]- u_jacobi[-3]))<0.0005:
                        #print("Jacobi vorzeitiges stop", len(u_jacobi))
                        convergenz = True
                        #print("convergenz_Newton",convergenz)
                        break
            #print(iter)

        if convergenz:
            """J1 = J1[1:, 1:]
            J2 = J2[1:, 1:]
            J3 = J3[1:, 1:]
            J4 = J4[1:, 1:]
            """

            dU_dQ = np.linalg.inv(J4 - np.dot(J3, np.dot(np.linalg.inv(J1), J2)))
            #dU_dQ_test = np.linalg.inv(J4 - np.dot(J3, np.dot(np.linalg.inv(J1), J2)))

            # evaluation
            if all(np.diag(dU_dQ) > 0):
                evaluation = 1
            else:

                evaluation = 0
            pass



    except Exception as e:
        print(f"Fehler in newtonrapson: {str(e)}")
    #print(abs(u_jacobi[-1, :]))
    print("u_jacobi", u_jacobi[-1,:])
    return u_jacobi[-1, :], evaluation, np.diag(dU_dQ)
