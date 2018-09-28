class Breakout():
    def __init__(self):
        self.env_name = 'BreakoutDeterministic-v4'
        self.start_score = 0
        self.start_lives = 5

    def get_ingame_action(self, action):
        if action == 0:
            return 1
        elif action == 1:
            return 2
        else:
            return 3
