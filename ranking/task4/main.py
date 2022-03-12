from ranking.task4.lambda_boosting import Solution


if __name__ == '__main__':
    solution = Solution()
    scores = solution.fit()
    print(scores)
