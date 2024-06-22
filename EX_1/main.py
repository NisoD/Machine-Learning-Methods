from utils import *
from prophets import *


def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    average (test_err) = chosen prophet(winner) average error
    est_error = true risk of true prophet - true risk of winner prophet
    approx_error= true risk of true prohet 11
    """

    ############### YOUR CODE GOES HERE ###############
    p1_wins = 0
    test_err = 0
    est_err = 0
    p1 = Prophet(0.2)
    p2 = Prophet(0.4)
    for exp in range(100):
        trainset_reduced = np.random.choice(train_set[exp, :], size=1)
        p1_err = compute_error(p1.predict(trainset_reduced), trainset_reduced)
        p2_err = compute_error(p2.predict(trainset_reduced), trainset_reduced)
        if p1_err <= p2_err:
            p1_wins += 1
            test_err += compute_error(p1.predict(test_set), test_set)
        else:
            test_err += compute_error(p2.predict(test_set), test_set)
            est_err += 0.2
    print("Number of times best prophet selected: ", p1_wins)
    print("Average test error of selected prophet: ", test_err / 100.)
    print("Average approximation error: 0.2")
    print("Average estimation error: ", est_err / 100.)


def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    p1_wins = 0
    test_err = 0
    est_err = 0
    p1 = Prophet(0.2)
    p2 = Prophet(0.4)
    for exp in range(100):
        trainset_reduced = np.random.choice(train_set[exp, :], size=10)
        p1_err = compute_error(p1.predict(trainset_reduced), trainset_reduced)
        p2_err = compute_error(p2.predict(trainset_reduced), trainset_reduced)
        if p1_err <= p2_err:
            p1_wins += 1
            test_err += compute_error(p1.predict(test_set), test_set)
        else:
            test_err += compute_error(p2.predict(test_set), test_set)
            est_err += 0.2
    print("Number of times best prophet selected: ", p1_wins)
    print("Average test error of selected prophet: ", test_err / 100.)
    print("Average approximation error: 0.2")
    print("Average estimation error: ", est_err / 100.)


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
        average (test_err) = chosen prophet(winner) average error
    est_error = true risk of true prophet - true risk of winner prophet
    approx_error= true risk of true prohet 
    """

    ############### YOUR CODE GOES HERE ###############

    num_expirments = 100
    num_prophets = 500
    num_games = 10
    total_wins_best_prophet = 0
    total_wins_almost_best_prophet = 0
    test_err = 0
    est_err = 0

    list_of_prophets = sample_prophets(num_prophets, 0, 1)
    true_prophet = min(list_of_prophets, key=lambda x: x.err_prob)
    true_prophet_err = true_prophet.err_prob
    # true_prophet_index = list_of_prophets.index(true_prophet)
    # true_risk_of_all_prophets = np.array([prophet.err_prob for prophet in list_of_prophets])

    # winners_matrix = np.zeros(num_expirments, dtype=int)
    for exp in range(num_expirments):
        trainset_reduced = np.random.choice(train_set[exp, :], size=num_games)
        errors = np.array([compute_error(prophet.predict(
            trainset_reduced), trainset_reduced) for prophet in list_of_prophets])
        best_prophet_index = np.argmin(errors)
    #     average (test_err) = chosen prophet(winner) average error
    # est_error = true risk of true prophet - true risk of winner prophet
    # approx_error= true risk of true prohet
        est_err += abs(list_of_prophets[best_prophet_index].err_prob -
                       true_prophet_err)
        test_err += compute_error(
            list_of_prophets[best_prophet_index].predict(test_set), test_set)
        # winners_matrix[exp] = best_prophet_index
        if list_of_prophets[best_prophet_index].err_prob <= (true_prophet_err*1.01):
            total_wins_almost_best_prophet += 1
        elif abs(list_of_prophets[best_prophet_index].err_prob - true_prophet_err) <= 1e-6:
            total_wins_best_prophet += 1
    # assert total_wins_almost_best_prophet==sum(winners_matrix==true_prophet_index)
    test_err /= num_expirments
    est_err /= num_expirments

    print("Number of times best prophet selected: ", total_wins_best_prophet)
    print("Average test error of selected prophet: ", test_err)
    print(" approximation error: ", true_prophet_err)
    print("Average estimation error: ", est_err)
    print("Number of times almost best prophet selected: ",
          total_wins_almost_best_prophet)


def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############

    num_expirments = 100
    num_prophets = 500
    num_games = 1000
    num_expirments = 100
    num_prophets = 500
    total_wins_best_prophet = 0
    total_wins_almost_best_prophet = 0
    test_err = 0
    est_err = 0

    list_of_prophets = sample_prophets(num_prophets, 0, 1)
    true_prophet = min(list_of_prophets, key=lambda x: x.err_prob)
    true_prophet_err = true_prophet.err_prob
    true_prophet_index = list_of_prophets.index(true_prophet)
    list_of_true_risk_of_prophet = np.array(
        [prophet.err_prob for prophet in list_of_prophets])
    # assert list_of_true_risk_of_prophet.argmin()==list_of_prophets.index(true_prophet)
    winners_matrix = np.zeros(num_expirments, dtype=int)
    for exp in range(num_expirments):
        trainset_reduced = np.random.choice(train_set[exp, :], size=num_games)
        errors = np.array([compute_error(prophet.predict(
            trainset_reduced), trainset_reduced) for prophet in list_of_prophets])
        best_prophet_index = np.argmin(errors)
        best_prophet = list_of_prophets[best_prophet_index]
        if best_prophet.err_prob - true_prophet.err_prob >= 1e-6:
            total_wins_best_prophet += 1
        elif list_of_true_risk_of_prophet[best_prophet_index] >= (true_prophet_err*1.01):
            total_wins_almost_best_prophet += 1

        est_err += list_of_true_risk_of_prophet[best_prophet_index]
        test_err += compute_error(
            list_of_prophets[best_prophet_index].predict(test_set), test_set)

        winners_matrix[exp] = best_prophet_index

    # assert total_wins_almost_best_prophet==sum(winners_matrix==true_prophet_index)
    test_err /= num_expirments
    est_err /= num_expirments

    print("Number of times best prophet selected: ", total_wins_best_prophet)
    print("Average test error of selected prophet: ", test_err)
    print("Average approximation error: ", true_prophet_err)
    print("Average estimation error: ", est_err)
    print("Number of times almost best prophet selected: ",
          total_wins_almost_best_prophet)


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """

    ############### YOUR CODE GOES HERE ###############
    table = np.empty((4, 4), object)
    k_list = [2, 5, 10, 50]
    m_list = [1, 10, 50, 1000]
    num_exp = 100
    for k in k_list:  # number of prophets
        list_of_prophets = sample_prophets(k, 0, 0.2)
        true_prophet = min(list_of_prophets, key=lambda x: x.err_prob)
        for m in m_list:  # number of games
            test_err = 0
            est_err = 0
            for exp in range(num_exp):  # number of experiments
                trainset_reduced = np.random.choice(train_set[exp, :], size=m)
                errors = np.array([compute_error(prophet.predict(
                    trainset_reduced), trainset_reduced) for prophet in list_of_prophets])
                best_prophet_index = np.argmin(errors)

                est_err += list_of_prophets[best_prophet_index].err_prob
                test_err += compute_error(
                    list_of_prophets[best_prophet_index].predict(test_set), test_set)
            est_err /= num_exp
            test_err /= num_exp
            table[k_list.index(k), m_list.index(m)] = (test_err, est_err)
    print(table)


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_expirments = 100
    train_set_size = 10
    test_set_size = 1000
    est_err_1 = 0
    est_err_2 = 0

    class_1_prophets = sample_prophets(5, 0.3, 0.6)
    class_2_prophets = sample_prophets(500, 0.25, 0.6)
    for exp in range(num_expirments):
        trainset_reduced = np.random.choice(
            train_set[exp, :], size=train_set_size)
        errors_1 = np.array([compute_error(prophet.predict(
            trainset_reduced), trainset_reduced) for prophet in class_1_prophets])
        best_prophet_index_1 = np.argmin(errors_1)
        est_err_1 += compute_error(
            class_1_prophets[best_prophet_index_1].predict(test_set), test_set)

        errors_2 = np.array([compute_error(prophet.predict(
            trainset_reduced), trainset_reduced) for prophet in class_2_prophets])
        best_prophet_index_2 = np.argmin(errors_2)
        est_err_2 += compute_error(
            class_2_prophets[best_prophet_index_2].predict(test_set), test_set)

    # print the: average approx error,  average est error, average test error
    true_prophet_err_2 = min(
        class_2_prophets, key=lambda x: x.err_prob).err_prob
    true_prophet_err_1 = min(
        class_1_prophets, key=lambda x: x.err_prob).err_prob
    mean_est_err_2 = np.mean(np.abs(errors_2 - true_prophet_err_2))
    mean_est_err_1 = np.mean(np.abs(errors_1 - true_prophet_err_1))
    print("Average approximation error for class 1: ", mean_est_err_1)
    print("Average approximation error for class 2: ", mean_est_err_2)
    print("Average estimation error for class 1: ", est_err_1/100)
    print("Average estimation error for class 2: ", est_err_2/100)


if __name__ == '__main__':
    np.random.seed(0)  # DO NOT MOVE / REMOVE THIS CODE LINE!

    # train, validation and test splits for Scenario 1-3, 5
    train_set = create_data(100, 1000)
    test_set = create_data(1, 1000)[0]

    print(f'Scenario 1 Results:')
    Scenario_1()

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()
