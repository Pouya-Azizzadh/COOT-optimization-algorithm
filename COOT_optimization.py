import numpy as np

def COOT(N, Max_iter, lb, ub, dim, fobj):
    if np.isscalar(ub):
        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

    NLeader = int(0.1 * N)
    Ncoot = N - NLeader
    Convergence_curve = np.zeros(Max_iter)
    gBest = np.zeros(dim)
    gBestScore = float('inf')

    # Initialize the positions of Coots
    CootPos = np.random.rand(Ncoot, dim) * (ub - lb) + lb
    CootFitness = np.zeros(Ncoot)

    # Initialize the locations of Leaders
    LeaderPos = np.random.rand(NLeader, dim) * (ub - lb) + lb
    LeaderFit = np.zeros(NLeader)

    for i in range(Ncoot):
        
        CootFitness[i] = fobj(CootPos[i, :])['cost']
        if gBestScore > CootFitness[i]:
            gBestScore = CootFitness[i]
            gBest = CootPos[i, :]

    for i in range(NLeader):
        LeaderFit[i] = fobj(LeaderPos[i, :])['cost']
        if gBestScore > LeaderFit[i]:
            gBestScore = LeaderFit[i]
            gBest = LeaderPos[i, :]

    Convergence_curve[0] = gBestScore
    l = 1  # Loop counter

    while l < Max_iter:
        B = 2 - l * (1 / Max_iter)
        A = 1 - l * (1 / Max_iter)

        for i in range(Ncoot):
            if np.random.rand() < 0.5:
                R = -1 + 2 * np.random.rand()
                R1 = np.random.rand()
            else:
                R = -1 + 2 * np.random.rand(dim)
                R1 = np.random.rand(dim)

            k = (i % NLeader)
            if np.random.rand() < 0.5:
                CootPos[i, :] = 2 * R1 * np.cos(2 * np.pi * R) * (LeaderPos[k, :] - CootPos[i, :]) + LeaderPos[k, :]
                # Check boundaries
                CootPos[i, :] = np.clip(CootPos[i, :], lb, ub)
            else:
                if np.random.rand() < 0.5 and i != 0:
                    CootPos[i, :] = (CootPos[i, :] + CootPos[i - 1, :]) / 2
                else:
                    Q = np.random.rand(dim) * (ub - lb) + lb
                    CootPos[i, :] = CootPos[i, :] + A * R1 * (Q - CootPos[i, :])
                CootPos[i, :] = np.clip(CootPos[i, :], lb, ub)

        # Fitness of location of Coots
        for i in range(Ncoot):
            CootFitness[i] = fobj(CootPos[i, :])['cost']
            k = (i % NLeader)
            # Update the location of coot
            if CootFitness[i] < LeaderFit[k]:
                Temp = LeaderPos[k, :].copy()
                TemFit = LeaderFit[k]
                LeaderFit[k] = CootFitness[i]
                LeaderPos[k, :] = CootPos[i, :]
                CootFitness[i] = TemFit
                CootPos[i, :] = Temp

        # Fitness of location of Leaders
        for i in range(NLeader):
            if np.random.rand() < 0.5:
                R = -1 + 2 * np.random.rand()
                R3 = np.random.rand()
            else:
                R = -1 + 2 * np.random.rand(dim)
                R3 = np.random.rand(dim)

            if np.random.rand() < 0.5:
                Temp = B * R3 * np.cos(2 * np.pi * R) * (gBest - LeaderPos[i, :]) + gBest
            else:
                Temp = B * R3 * np.cos(2 * np.pi * R) * (gBest - LeaderPos[i, :]) - gBest

            Temp = np.clip(Temp, lb, ub)
            TempFit = fobj(Temp)['cost']
            # Update the location of Leader
            if gBestScore > TempFit:
                LeaderFit[i] = gBestScore
                LeaderPos[i, :] = gBest
                gBestScore = TempFit
                gBest = Temp

        Convergence_curve[l] = gBestScore
        l += 1
    result=fobj(gBest)
    return Convergence_curve, gBest, gBestScore,result


