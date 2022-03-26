from ranking.task5.knrm import Solution
from ranking.task5.params import PATH_GLOVE, PATH_GLUE_QQP


def run_solution():
    solution = Solution(PATH_GLUE_QQP, PATH_GLOVE)
    solution.train(10)


if __name__ == '__main__':
    run_solution()
