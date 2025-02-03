import numpy as np

def e_step(alpha, beta, transition_matrix, emission_matrix, observed_sequence):
    T, num_states = alpha.shape
    posterior = np.zeros((T, num_states))

    for t in range(T):
        for i in range(num_states):
            posterior[t, i] = (alpha[t, i] * beta[t, i]) / np.sum(alpha[t, :] * beta[t, :])
    
    return posterior

def forward_backward_algorithm(transition_matrix, emission_matrix, observed_sequence):
    num_states = transition_matrix.shape[0]
    T = len(observed_sequence)

    alpha = np.zeros((T, num_states))
    beta = np.zeros((T, num_states))

    alpha[0, :] = emission_matrix[:, observed_sequence[0]] / np.sum(emission_matrix[:, observed_sequence[0]])

    for t in range(1, T):
        for j in range(num_states):
            alpha[t, j] = np.dot(alpha[t - 1, :], transition_matrix[:, j]) * emission_matrix[j, observed_sequence[t]]

    beta[T - 1, :] = 1

    for t in range(T - 2, -1, -1):
        for i in range(num_states):
            beta[t, i] = np.dot(transition_matrix[i, :], beta[t + 1, :] * emission_matrix[:, observed_sequence[t + 1]])

    return alpha, beta

def m_step(posterior, observed_sequence, transition_matrix, emission_matrix):
    num_states = transition_matrix.shape[0]
    T = len(observed_sequence)

    transition_counts = np.zeros((num_states, num_states))
    for t in range(T - 1):
        for i in range(num_states):
            for j in range(num_states):
                transition_counts[i, j] += posterior[t, i] * transition_matrix[i, j] * emission_matrix[j, observed_sequence[t+1]]

    transition_matrix_updated = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)

    emission_counts = np.zeros((num_states, 2))  # assuming two observations: Good Mood = 0, Bad Mood = 1
    for t in range(T):
        for i in range(num_states):
            emission_counts[i, observed_sequence[t]] += posterior[t, i]

    emission_matrix_updated = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)

    return transition_matrix_updated, emission_matrix_updated

def em_algorithm(observed_sequence, initial_theta, max_iter=100, tol=1e-6):
    transition_matrix = np.array([[1 - initial_theta, initial_theta], [initial_theta, 1 - initial_theta]])
    emission_matrix = np.array([[initial_theta / 2, 1 - initial_theta / 2], [initial_theta, 1 - initial_theta]])

    for iteration in range(max_iter):
        alpha, beta = forward_backward_algorithm(transition_matrix, emission_matrix, observed_sequence)
        
        posterior = e_step(alpha, beta, transition_matrix, emission_matrix, observed_sequence)

        transition_matrix_updated, emission_matrix_updated = m_step(posterior, observed_sequence, transition_matrix, emission_matrix)

        diff_transition = np.linalg.norm(transition_matrix_updated - transition_matrix)
        diff_emission = np.linalg.norm(emission_matrix_updated - emission_matrix)
        
        transition_matrix = transition_matrix_updated
        emission_matrix = emission_matrix_updated

        log_likelihood = np.sum(np.log(np.sum(alpha * beta, axis=1)))
        print(f"Iteration {iteration + 1}, Log-likelihood: {log_likelihood}")

        if diff_transition < tol and diff_emission < tol:
            print(f"Convergence reached at iteration {iteration + 1}")
            break

    return transition_matrix, emission_matrix

observed_sequence = [1, 1, 1, 0, 1]  

initial_theta = 0.5

# Run the EM algorithm
transition_matrix_final, emission_matrix_final = em_algorithm(observed_sequence, initial_theta)

print("Final Transition Matrix:")
print(transition_matrix_final)
print("Final Emission Matrix:")
print(emission_matrix_final)
