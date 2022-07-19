from abt.task7 import split


def generate_data(n_users: int):
    experiments = [{'experiment_id': 'exp_1', 'count_slots': 2, 'conflict_experiments': []},
                   {'experiment_id': 'exp_2', 'count_slots': 2, 'conflict_experiments': []},
                   {'experiment_id': 'exp_3', 'count_slots': 2, 'conflict_experiments': []}]
    users = [f'uid_{i}' for i in range(n_users)]
    return experiments, users


if __name__ == '__main__':
    experiments, users = generate_data(4)
    splitter = split.ABSplitter(5, 'salt10', 'salt2')
    splitter.split_experiments(experiments)
    print(splitter.slot_to_experiments)
    user_groups = [splitter.process_user(user) for user in users]
    print(user_groups)
