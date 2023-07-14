# This is where you can define the logic to chain together your models
class Chain:
    def __init__(self, models):
        self.models = models

    def run(self, input):
        for model in self.models:
            input = model.run(input)
        return input
