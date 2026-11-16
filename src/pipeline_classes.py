class PipelineStage:
    def __init__(self, title, frame):
        self.title = title
        self.frame = frame

class PipelineSection:
    def __init__(self, title):
        self.title = title
        self.stages = []

    def add_stage(self, title, frame):
        self.stages.append(PipelineStage(title, frame));