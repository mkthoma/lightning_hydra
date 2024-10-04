# Dog Breed Classifier using PyTorch Lightning, Hydra and Docker

This project is a dog breed classifier that uses PyTorch Lightning, Hydra and Docker. It is a simple example of how to use these technologies to build a machine learning model. It is not meant to be a production-ready application.

It is an adaptation of the [my earlier repository](https://github.com/mkthoma/pytorch_lightning_docker) to showcase the use of Hydra for configuration management and Docker for containerization.

Before you start, make sure you have the following:


1. ```
    touch .project-root
    ```

2.  ```
    export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
    ```
3.  ```
    uv sync --index-strategy unsafe-best-match
    ```
4.  ```
    echo 'export KAGGLE_USERNAME="your_username"' >> ~/.bashrc
    echo 'export KAGGLE_KEY="your_api_key"' >> ~/.bashrc
    source ~/.bashrc
    ```

## Docker commands
Based on the provided Dockerfile, the Docker commands for building the image and running train, eval, and infer with volume mounts for model artifacts. Here are the commands:

1. Build the Docker image:

    ```
    docker build -t dogbreed-classifier .
    ```

2. Run the train command:

    ```
    docker run --rm dogbreed-classifier train
    ```

3.  Run the eval command:

    ```
    docker run --rm dogbreed-classifier eval
    ``` 

4. Run the infer command:

    ```
    docker run --rm -v $(pwd)/outputs:/app/outputs dogbreed-classifier infer
    ```

