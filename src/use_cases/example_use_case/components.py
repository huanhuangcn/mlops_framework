class ExampleComponent:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def execute(self, input_data):
        # Implement the logic for this component
        processed_data = self.process(input_data)
        return processed_data

    def process(self, input_data):
        # Placeholder for processing logic
        return input_data  # Modify this as per the actual processing needed

def another_component_function(data):
    # Another component function that can be reused
    return data  # Modify this as per the actual logic needed