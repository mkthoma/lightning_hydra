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
    docker run --rm -v $(pwd)/logs:/app/logs -v $(pwd)/data:/app/data dogbreed-classifier src/train.py
    ```

3.  Run the eval command:

    ```
    docker run --rm -v $(pwd)/logs:/app/logs -v $(pwd)/data:/app/data dogbreed-classifier src/eval.py
    ``` 

4. Run the infer command:

    ```
    docker run --rm -v $(pwd)/model_artifacts:/app/model_artifacts -v $(pwd)/logs:/app/logs -v $(pwd)/data:/app/data dogbreed-classifier src/infer.py 
    ```

## Testing

This project uses pytest for unit testing and code coverage analysis. The tests are located in the `tests` folder.

### Running Tests

To run the tests, use the following command:
```
pytest tests/
```
To run individual tests, you can use the following command:
```
pytest tests/test_train.py
```

```
pytest tests/test_eval.py
```

```
pytest tests/test_infer.py
```

```
pytest tests/test_models/test_timm_classifier.py
```

```
pytest tests/test_datamodules/test_dogbreed.py
```

```
pytest tests/test_models/test_timm_classifier.py
```
## Code Coverage

To generate the code coverage report, use the following command:

```
coverage run -m pytest
coverage report -m
```

![image](https://github.com/user-attachments/assets/9bcea1c0-d25c-46a8-86bb-820cca1bee3f)


The CodeCov report can be found [here](https://app.codecov.io/github/mkthoma/lightning_hydra).

