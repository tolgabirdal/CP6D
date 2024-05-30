1. Compute Non-Conformity Score --> arccos(q1.T q2) --> Non-conformity Score
2. Given Non-conformity Scores on calibration set, compute non-conformity score for given test poses.
3. Get prediction region with top 5 p_values
4. Gaussian Entropy for Translation && Bingham Entropy for Rotation