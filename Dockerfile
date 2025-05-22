# Use a base image with Miniconda pre-installed
FROM continuumio/miniconda3:latest
# For GPU support or more ML-specific pre-installed libraries, consider:
# FROM gcr.io/deeplearning-platform-release/base-cpu   # For CPU
# FROM gcr.io/deeplearning-platform-release/base-gpu   # For GPU (requires NVIDIA drivers on host for local build/run)

# Set environment variables to ensure Conda is in PATH for non-interactive shells
ENV PATH /opt/conda/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Copy the Conda environment file
COPY environment.yml .

# Create the Conda environment from the .yml file
# Using mamba for faster environment creation (optional, but recommended)
# First, install mamba in the base conda environment
RUN conda install --yes --quiet mamba -n base -c conda-forge && \
    mamba env create -f environment.yml && \
    # Clean up Conda caches to reduce image size
    conda clean --all --yes --quiet && \
    mamba clean --all --yes --quiet

# Copy your MLOps framework source code into the container
# Adjust these COPY commands based on your project structure
COPY src/ /app/src/
COPY config/ /app/config/
# COPY data/ /app/data/ # Only copy if small and needed during build/runtime directly
# COPY scripts/ /app/scripts/ # If you have utility scripts

# Activate the Conda environment for subsequent RUN, CMD, ENTRYPOINT commands
# The SHELL command changes the shell used for executing commands.
# This form ensures that commands are run within the activated conda environment.
SHELL ["conda", "run", "-n", "mlops_env", "/bin/bash", "-c"]

# Example: Verify the environment is active and Python version
RUN echo "Conda environment 'mlops_env' is active." && \
    python --version && \
    pip list

# Define the command to run your application (replace with your actual command)
# This is just an example. For Vertex AI Custom Training, the ENTRYPOINT
# might be your training script. For a serving container, it might start a web server.
# ENTRYPOINT ["python", "src/main_script.py"]
# Or for a training job:
# ENTRYPOINT ["python", "src/use_cases/example_use_case/pipeline.py"]
# CMD ["--default-arg", "value"]

# If this container is for serving a model, expose the port
# EXPOSE 8080