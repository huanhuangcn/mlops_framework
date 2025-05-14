def validate_data(data):
    """
    Validates the input data for the machine learning pipeline.

    Args:
        data (dict): The input data to validate.

    Returns:
        bool: True if the data is valid, False otherwise.
    """
    # Example validation logic
    if not isinstance(data, dict):
        print("Data should be a dictionary.")
        return False

    required_keys = ['feature1', 'feature2', 'label']
    for key in required_keys:
        if key not in data:
            print(f"Missing required key: {key}")
            return False

    # Add more validation rules as needed

    return True


def validate_data_schema(data, schema):
    """
    Validates the input data against a predefined schema.

    Args:
        data (dict): The input data to validate.
        schema (dict): The schema to validate against.

    Returns:
        bool: True if the data matches the schema, False otherwise.
    """
    # Example schema validation logic
    for key, expected_type in schema.items():
        if key not in data:
            print(f"Missing key: {key}")
            return False
        if not isinstance(data[key], expected_type):
            print(f"Key {key} should be of type {expected_type.__name__}.")
            return False

    return True