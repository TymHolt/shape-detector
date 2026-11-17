import cv2

def clamp_index(curr_value, min_value, max_value):
    return max(min_value, min(max_value, curr_value))

class PipelineAdjuster:
    def __init__(self, title, lower, initial, upper, step):
        self.title = title
        self.lower = lower
        self.current = initial
        self.upper = upper
        self.step = step

    def clamp_current(self):
        self.current = clamp_index(self.current, self.lower, self.upper)

class PipelineStage:
    def __init__(self, title, frame):
        self.title = title
        self.frame = frame
        self.adjusters = []

    def add_adjuster(self, title, lower, initial, upper, step):
        for adjuster in self.adjusters:
            if adjuster.title == title:
                adjuster.lower = lower
                adjuster.initial = initial
                adjuster.upper = upper
                adjuster.step = step
                return adjuster

        adjuster = PipelineAdjuster(title, lower, initial, upper, step)
        self.adjusters.append(adjuster)
        return adjuster

class PipelineSection:
    def __init__(self, title):
        self.title = title
        self.stages = []

    def add_stage(self, title, frame):
        for stage in self.stages:
            if stage.title == title:
                stage.frame = frame
                return stage
            
        stage = PipelineStage(title, frame)
        self.stages.append(stage)
        return stage

SELECTION_INDEX_MAX = 1000

VIEW_DEPTH_MIN = 0
VIEW_DEPTH_SECTION = 0
VIEW_DEPTH_STAGE = 1
VIEW_DEPTH_ADJUSTER = 2
VIEW_DEPTH_MAX = 2

INPUT_AXIS_VERTICAL = 0
INPUT_AXIS_HORIZONTAL = 1
INPUT_AXIS_CONTROL = 3

GUI_DARK = (0, 150, 0)
GUI_LIGHT = (100, 255, 100)

class PipelineGui:
    def __init__(self):
        self.visible = False
        self.sections = []
        self.view_depth = VIEW_DEPTH_SECTION

        # Default value, gets clamped to max for final view
        self.section_index = SELECTION_INDEX_MAX
        self.stage_index = SELECTION_INDEX_MAX
        self.adjuster_index = SELECTION_INDEX_MAX

    def add_section(self, title):
        for section in self.sections:
            if section.title == title:
                return section
    
        section = PipelineSection(title)
        self.sections.append(section)
        return section

    def get_current_section(self):
        count = len(self.sections)
        if count <= self.section_index or count < 1:
            return None
        return self.sections[self.section_index]

    def get_current_stage(self):
        current_section = self.get_current_section()
        if current_section == None:
            return None        

        count = len(current_section.stages)
        if count <= self.stage_index or count < 1:
            return None                
        return current_section.stages[self.stage_index]

    def get_current_adjuster(self):
        current_stage = self.get_current_stage()
        if current_stage == None:
            return None

        count = len(current_stage.adjusters)
        if count <= self.adjuster_index or count < 1:
            return None
        return current_stage.adjusters[self.adjuster_index]
    
    def clamp_all_indices(self):
        self.view_depth = clamp_index(self.view_depth, VIEW_DEPTH_MIN, VIEW_DEPTH_MAX)
        
        self.section_index = clamp_index(self.section_index, 0, len(self.sections) - 1)
        current_section = self.get_current_section()
        if current_section == None:
            self.stage_index = SELECTION_INDEX_MAX
            self.adjuster_index = 0
            return

        self.stage_index = clamp_index(self.stage_index, 0, len(current_section.stages) - 1)
        current_stage = self.get_current_stage()
        if current_stage == None:
            self.adjuster_index = 0
            return

        self.adjuster_index = clamp_index(self.adjuster_index, 0, len(current_stage.adjusters) - 1)

    def handle_input(self, axis, direction):
        if not self.visible:
            return

        if axis == INPUT_AXIS_VERTICAL:
            if self.view_depth == VIEW_DEPTH_SECTION:
                self.section_index += direction
                self.stage_index = SELECTION_INDEX_MAX # Always show last stage of section
            elif self.view_depth == VIEW_DEPTH_STAGE:
                self.stage_index += direction
            elif self.view_depth == VIEW_DEPTH_ADJUSTER:
                self.adjuster_index += direction
        elif axis == INPUT_AXIS_HORIZONTAL:
            self.view_depth += direction
        elif axis == INPUT_AXIS_CONTROL:
            if self.view_depth == VIEW_DEPTH_ADJUSTER:
                current_adjuster = self.get_current_adjuster()
                if current_adjuster != None:
                    current_adjuster.current += current_adjuster.step * direction
                    current_adjuster.clamp_current()

    def render(self, frame):
        current_section = self.get_current_section()
        current_stage = self.get_current_stage()
        current_adjuster = self.get_current_adjuster()

        if current_section == None or current_stage == None or not self.visible:
            return 
    
        if self.view_depth == VIEW_DEPTH_SECTION:
            y = 30
            for section in self.sections:
                text = f"{section.title} >"
                text_color = GUI_DARK
                if section.title == current_section.title:
                    text_color = GUI_LIGHT
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                y += 30
        elif self.view_depth == VIEW_DEPTH_STAGE:
            cv2.putText(frame, f"< {current_section.title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GUI_DARK, 2, cv2.LINE_AA)           
            y = 60
            for stage in current_section.stages:
                text = f"{stage.title}"
                text_color = GUI_DARK
                if stage.title == current_stage.title:
                    text_color = GUI_LIGHT
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                y += 30
        elif self.view_depth == VIEW_DEPTH_ADJUSTER:
            cv2.putText(frame, f"< {current_stage.title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GUI_DARK, 2, cv2.LINE_AA)           
            y = 60
            for adjuster in current_stage.adjusters:
                text = f"{adjuster.title} {adjuster.current}"
                text_color = GUI_DARK
                if current_adjuster != None and adjuster.title == current_adjuster.title:
                    text_color = GUI_LIGHT
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                y += 30
                

