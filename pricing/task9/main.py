from pricing.task9.multiarmed import BernoulliEnv, calculate_regret
from pricing.task9.strategies import EpsGreedy, UCB1, Thompson


def main():
    be = BernoulliEnv([0.3, 0.5, 0.7])
    eps_1 = EpsGreedy(be.n_arms, 0.1)
    eps_2 = EpsGreedy(be.n_arms, 0.3)
    eps_3 = EpsGreedy(be.n_arms, 0.5)
    ucb = UCB1(be.n_arms)
    thompson = Thompson(be.n_arms)

    eps_regrets = calculate_regret(be, eps_1)
    eps_2_regrets = calculate_regret(be, eps_2)
    eps_3_regrets = calculate_regret(be, eps_3)
    ucb_regrets = calculate_regret(be, ucb)
    thompson_regrets = calculate_regret(be, thompson)

    print(f'eps1_regrets: {eps_regrets}')
    print(f'eps_2_regrets: {eps_2_regrets}')
    print(f'eps_3_regrets: {eps_3_regrets}')
    print(f'ucb_regrets: {ucb_regrets}')
    print(f'thompson_regrets: {thompson_regrets}')


if __name__ == '__main__':
    main()
